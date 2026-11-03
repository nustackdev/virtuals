#!lua name=nu_notify_v1
-- Server-side notify routing for virtuals RedisPublisher / RedisObserver.
--
-- Lua does no verification and no number handling: Python encodes every key
-- segment and every routing hint (keyenc.py) and Lua only compares and
-- concatenates opaque strings. Python's Filter.matches stays the only exact
-- matcher; over-delivery is dropped by the observer.
--
-- Index layout (P = channel prefix, nx1 = versioned namespace):
--   P:nx1:p:<d>:<enc>      SET of hashes, prefix depth d      P:nx1:pc HASH d -> count
--   P:nx1:s:<d>:<enc>      SET of hashes, suffix depth d      P:nx1:sc HASH d -> count
--   P:nx1:l:<n>:           SET of hashes, key length n
--   P:nx1:w:<n:pos>:<enc>  SET of hashes, wildcard signature  P:nx1:wc HASH n:pos -> count
--   P:nx1:a                SET of hashes matching every key
--   P:nx1:refs             HASH h -> cmsgpack of {kind, field, enc} triples
--
-- Handlers take (argv, o): argv[o] is the first real argument. Function mode
-- passes o = 1, script (EVALSHA) mode passes o = 2 (argv[1] is the name).

local NS = ':nx1:'
local COUNTER = {p = 'pc', s = 'sc', w = 'wc'}

local function set_key(P, kind, field, enc)
  if kind == 'a' then return P .. NS .. 'a' end
  return P .. NS .. kind .. ':' .. field .. ':' .. enc
end

local function unreg(P, h)
  local refs_key = P .. NS .. 'refs'
  local packed = redis.call('HGET', refs_key, h)
  if not packed then return end
  local entries = cmsgpack.unpack(packed)
  for i = 1, #entries, 3 do
    local kind, field = entries[i], entries[i + 1]
    redis.call('SREM', set_key(P, kind, field, entries[i + 2]), h)
    local c = COUNTER[kind]
    if c then
      local ck = P .. NS .. c
      if redis.call('HINCRBY', ck, field, -1) <= 0 then redis.call('HDEL', ck, field) end
    end
  end
  redis.call('HDEL', refs_key, h)
end

-- nu1_reg ARGV: P, h, kind1, field1, enc1, ... -> 1 if newly indexed, else 0
local function reg(a, o)
  local P, h = a[o], a[o + 1]
  local refs_key = P .. NS .. 'refs'
  if redis.call('HEXISTS', refs_key, h) == 1 then return 0 end
  local entries = {}
  for i = o + 2, #a, 3 do
    local kind, field, enc = a[i], a[i + 1], a[i + 2]
    redis.call('SADD', set_key(P, kind, field, enc), h)
    local c = COUNTER[kind]
    if c then redis.call('HINCRBY', P .. NS .. c, field, 1) end
    entries[#entries + 1] = kind
    entries[#entries + 1] = field
    entries[#entries + 1] = enc
  end
  redis.call('HSET', refs_key, h, cmsgpack.pack(entries))
  return 1
end

local function arr_hdr(n)
  if n < 16 then return string.char(0x90 + n) end
  if n < 65536 then return '\220' .. string.char(math.floor(n / 256), n % 256) end
  return '\221' .. string.char(
    math.floor(n / 16777216) % 256, math.floor(n / 65536) % 256,
    math.floor(n / 256) % 256, n % 256)
end

local function depths(P, c)
  local out = {}
  for _, d in ipairs(redis.call('HKEYS', P .. NS .. c)) do out[#out + 1] = tonumber(d) end
  return out
end

-- nu1_route ARGV: P, hdr, packed1, nseg1, seg..., packed2, nseg2, seg..., ...
-- -> number of channels published
local function route(a, o)
  local P, hdr = a[o], a[o + 1]
  local pre = P .. NS
  local pd, sd = depths(P, 'pc'), depths(P, 'sc')
  local wsig = {}  -- n -> list of {field, {1-based positions}}
  for _, f in ipairs(redis.call('HKEYS', pre .. 'wc')) do
    local n, plist = string.match(f, '^(%d+):(.*)$')
    n = tonumber(n)
    local ps = {}
    for p in string.gmatch(plist, '%d+') do ps[#ps + 1] = tonumber(p) + 1 end
    wsig[n] = wsig[n] or {}
    table.insert(wsig[n], {f, ps})
  end
  local all = redis.call('SMEMBERS', pre .. 'a')

  local out, order = {}, {}
  local i, last = o + 2, #a
  while i <= last do
    local packed, n = a[i], tonumber(a[i + 1])
    local s = {}
    for k = 1, n do s[k] = a[i + 1 + k] end
    i = i + 2 + n

    local cand, list = {}, {}
    local function add(hs)
      for _, h in ipairs(hs) do
        if not cand[h] then cand[h] = true; list[#list + 1] = h end
      end
    end
    for _, d in ipairs(pd) do
      if d <= n then add(redis.call('SMEMBERS', pre .. 'p:' .. d .. ':' .. table.concat(s, '', 1, d))) end
    end
    for _, d in ipairs(sd) do
      if d <= n then add(redis.call('SMEMBERS', pre .. 's:' .. d .. ':' .. table.concat(s, '', n - d + 1, n))) end
    end
    add(redis.call('SMEMBERS', pre .. 'l:' .. n .. ':'))
    local ws = wsig[n]
    if ws then
      for _, w in ipairs(ws) do
        local fx = {}
        for _, p in ipairs(w[2]) do fx[#fx + 1] = s[p] end
        add(redis.call('SMEMBERS', pre .. 'w:' .. w[1] .. ':' .. table.concat(fx)))
      end
    end
    add(all)

    for _, h in ipairs(list) do
      local ks = out[h]
      if not ks then ks = {}; out[h] = ks; order[#order + 1] = h end
      ks[#ks + 1] = packed
    end
  end

  local sent = 0
  for _, h in ipairs(order) do
    local ks = out[h]
    local r = redis.call('PUBLISH', P .. ':notif:' .. h, hdr .. arr_hdr(#ks) .. table.concat(ks))
    if r == 0 then unreg(P, h) else sent = sent + 1 end
  end
  return sent
end

-- @@register (everything below is replaced by the EVALSHA footer in script mode)
redis.register_function('nu1_reg', function(_, args) return reg(args, 1) end)
redis.register_function('nu1_route', function(_, args) return route(args, 1) end)
