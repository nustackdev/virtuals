# Virtuals — Philosophy

## Views Are Infinite

A view is a lens over storage. It interprets raw keys as structure.

Dict view sees `{"a": 1, "b": 2}`. List view sees `[1, 2, 3]`. Set view sees `{1, 2, 3}`. But these are just the obvious ones.

A view can see a B-tree. A view can see a time-series with automatic bucketing. A view can see a graph with adjacency encoded in key prefixes. A view can see a sparse tensor.

The view protocol is fixed. What views can represent is not. Any structure that can be projected onto key-value pairs can have a view. Any optimization that can be expressed in how keys are laid out can be encoded.

This is the extensibility that matters. Not plugin architectures. Not configuration options. The ability to define entirely new structural interpretations without changing anything above or below.

## Lazy by Construction

Nothing loads until asked. Accessing `users["alice"]` doesn't materialize the entire dict — it reaches into storage for exactly that key. Iterating yields one item at a time from the backend.

This separation means the system can defer work until actually needed. Batching, caching, reordering — these become possible because the views don't eagerly consume.

## What Flows Through

Define a new view. It works with existing storage, existing containers, existing backends.

Define a new storage backend. It works with existing everything.

The protocols are the stable points. Implementations flow through them.

## The Stack

```
View       — structure interpretation
Container  — hierarchy maintenance
Storage    — byte persistence
```

Each layer depends only on the layer below. Each layer is independently replaceable. The contracts between layers are thin — just protocols.

This isn't modular design for its own sake. It's recognition that these concerns are orthogonal. How you interpret structure shouldn't know how you persist bytes. How you enforce hierarchy shouldn't know what "dict" means.
