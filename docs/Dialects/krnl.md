<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `krnl.acos` (::mlir::KrnlAcosOp)

Krnl acos scalar operation

Krnl acos scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`out` | floating-point

### `krnl.acosh` (::mlir::KrnlAcoshOp)

Krnl acosh scalar operation

Krnl acosh scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`out` | floating-point

### `krnl.asin` (::mlir::KrnlAsinOp)

Krnl asin scalar operation

Krnl asin scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`out` | floating-point

### `krnl.asinh` (::mlir::KrnlAsinhOp)

Krnl asinh scalar operation

Krnl asinh scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`out` | floating-point

### `krnl.atan` (::mlir::KrnlAtanOp)

Krnl atan scalar operation

Krnl atan scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`out` | floating-point

### `krnl.atanh` (::mlir::KrnlAtanhOp)

Krnl atanh scalar operation

Krnl atanh scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`out` | floating-point

### `krnl.block` (::mlir::KrnlBlockOp)

Krnl block operation


Syntax:

```
operation ::= `krnl.block` $loop $tile_size attr-dict `:` functional-type($loop, results)
```

Block a single for loop by a constant tile size. For instance,
$ib, $il = krnl.block %i, 4
means to block the for loop referred to by %i using a tile size of 4.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`tile_size` | ::mlir::IntegerAttr | 64-bit signless integer attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`loop` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
`loop_block` | any type
`loop_local` | any type

### `krnl.copy_from_tile_buffer` (::mlir::KrnlCopyFromBufferOp)

Copy from buffer.


Syntax:

```
operation ::= `krnl.copy_from_tile_buffer` $buffer `,` $dest `[` $starts `]`  attr-dict `:` type($buffer) `,` type($dest)
```

Operation that copy a destination memory from a buffer memory.
Starts indicate where the buffer data starts to go into the destination
memory. Start values must be at multiples of buffer size in all dimensions.
The buffer rank and dimensions are compile time constants.

If the buffer was oversized with respect of the actual data contained
in the tile, the actual tile size can be given using the tileSize
optional attribute. This attributes has the same rank as the buffer size,
and each dimension must be smaller or equal to the actual buffer size.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`tileSize` | ::mlir::ArrayAttr | 64-bit integer array attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`buffer` | memref of any type values
`dest` | memref of any type values
`starts` | index

### `krnl.copy_to_tile_buffer` (::mlir::KrnlCopyToBufferOp)

Copy to buffer.


Syntax:

```
operation ::= `krnl.copy_to_tile_buffer` $buffer `,` $source `[` $starts `]` `,`  $padValue  attr-dict
              `:` type($buffer) `,` type($source)
```

Operation that copy a source memory to a buffer memory.
Starts indicate where the source data starts to come from within
the source memory. Start values must be at multiples of buffer size
in all dimensions. The buffer rank and dimensions are compile time
constants.

The buffer will be entirely filled with the source data. By default,
the amount of data to copy is given by the size of the buffer.
In some cases, we may want to oversize a buffer for better cache,
simd, or loop unroll and jam reasons. If that is the case, the
actual tile size of the data to be copied over is given by an
optional tileSize attribute. This attributes has the same rank as
the buffer size, and each dimension must be smaller or equal to
the actual buffer size.

If there is not enough data in the source memory to fill the buffer,
because the operation reaches the upper bounds of the source memory,
several actions may happen.

* If padToNext attribute is given, the pad value will be copied from
  the last source data of to the next index for which index modulo padToNext
  is zero, i.e. to the end of a "cache line" of side padToLine. Pad
  of 1 means no padding, pad of buffer size means fully pad the buffer.
  Default is no padding (1). PadValue is used to initialized the padded
  areas.

* If overreadToNext attribute is given, the copy may read source past
  its upperbound value. This enable optimized code, e.g. using SIMD
  read operations even if going past the last value of the source
  memory, or unrolling and jaming copy loops to reduce memory latency.
  overreadToNext is expressed like padToNext: value of 1 means no
  reading past boundary; value of buffer size enables reading
  as many additional sourve value as needed to fill the full
  buffer. Default is buffer-size.

padToNext and overreadToNext are of the same rank as source and memory
memrefs.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`tileSize` | ::mlir::ArrayAttr | 64-bit integer array attribute
`padToNext` | ::mlir::ArrayAttr | 64-bit integer array attribute
`transpose` | ::mlir::BoolAttr | bool attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`buffer` | memref of any type values
`source` | memref of any type values
`starts` | index
`padValue` | any type

### `krnl.define_loops` (::mlir::KrnlDefineLoopsOp)

define_loops operation

The "krnl.define_loops" operation is used to define input loops,
those are the for loops appearing in the input program that we
intend to optimize.

#### Results:

| Result | Description |
| :----: | ----------- |
&laquo;unnamed&raquo; | any type

### `krnl.dim` (::mlir::KrnlDimOp)

Krnl dimensions operation.

Emits the dimension of a MemRef independent of the MemRef alloc:

"krnl.dim"(%memref, %index)

The index identifies the dimension within the shape which is going to be emitted.
Initially the krnl.dim operation depends on the alloc of the MemRef.
Unlike the std.dim operation which maintains a dependency on the alloc of the MemRef, the dimension emitted by krnl.dim will not depend on the alloc operation of the MemRef once the krnl.dim operation is lowered.

Any changes to the original MemRef size after the krnl.dim has been lowered will not be picked up by the emitted dimension. This allows the original MemRef to be safely modified via code transformations or affine map normalization without the risk of changing the value already emitted via krnl.dim.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`alloc` | memref of any type values
`index` | index

#### Results:

| Result | Description |
| :----: | ----------- |
`dimension` | index

### `krnl.dummy_cast` (::mlir::KrnlDummyCastOp)

A dummy Krnl operation to perform type casting.


Syntax:

```
operation ::= `krnl.dummy_cast` $in attr-dict `:` functional-type($in, results)
```

Krnl operation to perform dummy type casting to remove the type
dependencies existing between lowering of multiple IR objects.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in` | memref of any type values or tensor of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
`out` | memref of any type values or tensor of any type values

### `krnl.entry_point` (::mlir::KrnlEntryPointOp)

Indicate ONNX entry point

The "krnl.entry_point" function indicates the main entry
                           point of ONNX model.
### `krnl.erf` (::mlir::KrnlErfOp)

Krnl erf scalar operation

Krnl erf scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`out` | floating-point

### `krnl.get_induction_var_value` (::mlir::KrnlGetInductionVariableValueOp)

Krnl 


Syntax:

```
operation ::= `krnl.get_induction_var_value` `(` $loops `)` attr-dict `:` functional-type($loops, results)
```

Krnl operation to convert loop references to corresponding induction
variable values. This is useful for accessing optimized loop induction
variables, as they are not otherwise accessible during Krnl Dialect.

For example, this operation can be applied to loop references corresponding to
inter-tile iterations. The return values will be the starting index of the
current tile being iterated over.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`loops` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
`ind_var_vals` | any type

### `krnl.getref` (::mlir::KrnlGetRefOp)

Krnl a MemRef from within another MemRef starting at a specific offset.

Retreieves a MemRef from within another MemRef:

"krnl.getref"(%memref, %offset)

The offset is an integer which is used as an index into the input MemRef. It works
just like an array index.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`mempool` | memref of any type values
`offset` | integer
`value` | index

#### Results:

| Result | Description |
| :----: | ----------- |
`output` | memref of any type values

### `krnl.global` (::mlir::KrnlGlobalOp)

Krnl global operation

Operation for holding global data values. A global constant can have a
meaningful name recorded as its `name` attribute. Its content is stored
in the `value` dense/opaque element attribute.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`shape` | ::mlir::Attribute | any attribute
`name` | ::mlir::StringAttr | string attribute
`value` | ::mlir::Attribute | any attribute
`offset` | ::mlir::IntegerAttr | 64-bit signless integer attribute
`alignment` | ::mlir::IntegerAttr | 64-bit signless integer attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`output` | memref of any type values

### `krnl.runtime_instrument` (::mlir::KrnlInstrumentOp)

instrumentation point.

Operation that invokes the runtime instrument utility.
May be used for gdb.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`opID` | ::mlir::IntegerAttr | 64-bit signless integer attribute
`tag` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### `krnl.iterate` (::mlir::KrnlIterateOp)

iterate operation

The "krnl.iterate" operation is conceptually equivalent to a nested for loops.

For instance, say we have the following two
%l0, %l1 = krnl.define_loops 2
%o0, %o1 = krnl.optimize_loops  {
    // Identity schedule.
    krnl.return_loops %l0, %l1
}

Then, consider the following krnl.iterate operation:
krnl.iterate (%o0, %o1) with (%l0 -> %i0 = 0 to 10, %l1 -> %i1 = 0 to 10) {
  // Some operations.
}

It is equivalent to:
for (i0 = 0; i0 < 10; i0++)
  for (i1 = 0; i1 < 10; i1++)
    // Some operations.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
&laquo;unnamed&raquo; | any type

### `krnl.load` (::mlir::KrnlLoadOp)

A Krnl operation to load data from the memref.


Syntax:

```
operation ::= `krnl.load` $memref `[` $indices `]` attr-dict `:` type($memref)
```

The `krnl.load` op reads an element from a memref specified by an index
list. The output of load is a new value with the same type as the elements
of the memref. The arity of indices is the rank of the memref (i.e., if the
memref loaded from is of rank 3, then 3 indices are required for the load
following the memref identifier).

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`memref` | memref of any type values
`indices` | index

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | any type

### `krnl.matmul` (::mlir::KrnlMatMulOp)

Matmul operation for a single pannel.


Syntax:

```
operation ::= `krnl.matmul` $A `[` $aMemStart `]` `,`
              $B `[` $bMemStart `]` `,`
              $C `[` $cMemStart `]` `,`
              `(` $loops `)` `,`
              `(` $iComputeStart `,` $jComputeStart `,` $kComputeStart `)` `,`
              `(` $iGlobalUB `,` $jGlobalUB `,` $kGlobalUB `)`
              attr-dict `:` type($A) `,` type($B)`,` type($C) `,` `(` type($loops) `)`
```

Perform a matrix multiplication A * B + C
for a small tile A * B + C of sizes
[IxK] * [KxJ] + [IxJ].

The i/j/k ComputeStarts indicate the global indices of the first element
of a tile to be computed in the original computations.
The i/j/k GlobalUBs indicate the upper bounds in the original computations.

We provide 3 buffers for matrix multipy: A, B, and C. For each buffer,
we indicate the global indices pointing the beginning of the buffer.
If no buffers are used, i.e. the computation starts directly in the orginal
memory, the global index is 0. If a buffer for A is used to put data into
it starting at indices [i1, k1], where i1 & k1 are the global indices in
the original computations, then aMemStart0 and aMemStart1 are i1 & k1,
respectively.

If the A, B, or C buffers are larger than the actual data tile they
contain (see copy_to_tile_buffer), then the actual tile size must be
given using an optional attribute: ATileSize, BTileSize, or CTileSize.
These optional tile size have a rank of 2, and their values must be
equal or smaller than their corresponding buffer memrefs.

If the computation are further tiled with respect to the size of the
buffers A, B, or C, then the actual computation tile is given by
the optional tile attribute computeTileSize. Its rank is 3, for the
I, J, and K dimension. The actual A, B, and C buffer tile size 
(possibly specified by the optional parameters) must be a multiple of
the I, J, and K computeTileSizes, in their respective
dimensions (A: IxK], B: [KxJ], C: [IxJ]).

Note that the buffers A, B, and C can be of higher dimensionality than
the traditional 2D mentioned up to now, because of broadcasting rules.
At this time, we only support broadcast of arrays having ranks of 2 or
more. Because of the broadcast rules, the higher dimenstions have a
constant index during one matrix multiply. These fixed indices are
given as prefix dimensions in the starting indices for A, B, and C
as described above. E.g. if A has a rank of 3, and B has a rank of 2,
the starting indices for A are [d, i1, k1] where i1 and k1 are as
above, and d is index pointing to the current instance of the IxK
A matrix to be computed. B start indices would be unchanged at [k1, j1].

Simdize is used to state if simdization is requested.
Unrolling is used to unroll and jam loops as warrented.

Below is an example calculating a matrix multiply with pre-zeroed
C matrix with the sizes below. 
%A: memref<40x60xf32>, %B: memref<60x80xf32>, %C: memref<40x80xf32>

// 3 tiled loops.
%ii, %jj, %kk = krnl.define_loops 3
%ib, %il = krnl.block %ii 10 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
%jb, %jl = krnl.block %jj 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
%kb, %kl = krnl.block %kk 10 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// 3 subtiles.
%ilb, %ill = krnl.block %il 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
%jlb, %jll = krnl.block %jl 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
%klb, %kll = krnl.block %kl 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// Permute.
krnl.permute(%ib, %ilb, %ill, %jb, %jlb, %jll, %kb, %klb, %kll) 
    [0, 3, 6, 1, 4, 7, 2, 5, 8] : 
    !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, 
    !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// Outer 2 for i, j.
krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 40, 
                             %jj -> %j = 0 to 80, 
                             %kk -> %k = 0 to 60) {
    %i1, %j1 = krnl.get_induction_var_value(%ib, %jb) : 
      (!krnl.loop,!krnl.loop) -> (index, index)
    // Fill C buffer.
    %Cbuff = alloca(): memref<10x8xf32>  // n x m_simd
    krnl.copy_to_tile_buffer %Cbuff, %C[%i1, %j1], %f0 : 
      memref<10x8xf32>, memref<40x80xf32>
    // Outer 1 for k.
    krnl.iterate(%kb) with () {
        %k1 = krnl.get_induction_var_value(%kb) : (!krnl.loop) -> (index)
        // Fill A and B buffer
        %Abuff = alloca(): memref<10x10xf32> // i x k
        %Bbuff = alloca(): memref<10x8xf32>  // k x j_simd     
        krnl.copy_to_tile_buffer %Abuff, %A[%i1, %k1], %f0 :
          memref<10x10xf32>, memref<40x60xf32>
        krnl.copy_to_tile_buffer %Bbuff, %B[%k1, %j1], %f0 :
          memref<10x8xf32>, memref<60x80xf32>

        // Inner iterations for subtiles.
        krnl.iterate(%ilb, %jlb, %klb) with () {
            %i2, %j2, %k2 = krnl.get_induction_var_value(%ilb, %jlb, %klb) :
            (!krnl.loop,!krnl.loop,!krnl.loop) -> (index,index,index)

            krnl.matmul %Abuff[%i1, %k1], %Bbuff[%k1, %j1], %Cbuff[%i1, %j1],
                (%ill, %jll, %kll), (%i2, %j2, %k2), (%c40, %c80, %c60)
                { computeTileSize=[5,4,5], simdize=false, unroll=false } :
                memref<10x10xf32>, memref<10x8xf32>, memref<10x8xf32>,
                (!krnl.loop,!krnl.loop,!krnl.loop)
        }
    }
    // Copy back the data into C.
    krnl.copy_from_tile_buffer %Cbuff, %C[%i1, %j1] :
      memref<10x8xf32>, memref<40x80xf32>
}

Note that code is simdized along the J dim (last dim of B and C matrices).
For simd to be enabled, the simdized flag must be set to true, and the 
following condition must be true:
1) The vector length is the second entry of (i, j, k) compute tile size.
   The vector length must be a compile time constant.
2) last dim of B & C mem ref must be constant and a multiple of the vector
   length.
By copying the original data B and result C into tiles of known compile time
sizes, any computations can be made to simdize.

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`computeTileSize` | ::mlir::ArrayAttr | 64-bit integer array attribute
`aTileSize` | ::mlir::ArrayAttr | 64-bit integer array attribute
`bTileSize` | ::mlir::ArrayAttr | 64-bit integer array attribute
`cTileSize` | ::mlir::ArrayAttr | 64-bit integer array attribute
`simdize` | ::mlir::BoolAttr | bool attribute
`unroll` | ::mlir::BoolAttr | bool attribute
`overcompute` | ::mlir::BoolAttr | bool attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`A` | memref of any type values
`aMemStart` | index
`B` | memref of any type values
`bMemStart` | index
`C` | memref of any type values
`cMemStart` | index
`loops` | any type
`iComputeStart` | index
`jComputeStart` | index
`kComputeStart` | index
`iGlobalUB` | index
`jGlobalUB` | index
`kGlobalUB` | index

### `krnl.memcpy` (::mlir::KrnlMemcpyOp)

Krnl memcpy operation

In the KRNL dialect the reshape op
doesn't generate a new memory entry and treats a reshape like a cast.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`dest` | memref of any type values
`src` | memref of any type values
`size` | integer

### `krnl.memset` (::mlir::KrnlMemsetOp)

Set buffer to a given value.


Syntax:

```
operation ::= `krnl.memset` $dest `,` $value attr-dict `:` type($dest)
```

Krnl operation that set buffer to a given value.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`dest` | memref of any type values
`value` | any type

### `krnl.movable` (::mlir::KrnlMovableOp)

Krnl movable operation


Syntax:

```
operation ::= `krnl.movable` $region attr-dict
```

Encapsulates a list of operations, which should be moved under a newly lowered
affine for operation eventually, but cannot presently because the destination
affine for operation is not materialized yet.

This operation is automatically generated by the lowering of Krnl to affine dialect
to assist with maintaining the relative positioning of loop and inner-loop statements.
This construct is particularly helpful, for example, for lowering statements that
are nested imperfectly between an "eager" and a "lazy" loop.

### `krnl.permute` (::mlir::KrnlPermuteOp)

Krnl permute operation


Syntax:

```
operation ::= `krnl.permute` `(` $loops `)` $map attr-dict `:` type($loops)
```

Permute a set of affine for loops using a specified permutation map.
The permutation map `map` should be constructed in such way that the
for loop referred to by the i-th operand to permute operation is sent
to the `map[i]`-th position.

For example, the following krnl dialect IR:
```
%ii, %jj, %kk = krnl.define_loops 3
krnl.permute(%ii, %jj, %kk) [1, 2, 0] : !krnl.loop, !krnl.loop, !krnl.loop
krnl.iterate (%ii, %jj, %kk) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20, %kk -> %k = 0 to 30) {}
```
will be lowered to:
```
// Referenced by %kk
affine.for %arg0 = 0 to 30 {
  // Referenced by %ii
  affine.for %arg1 = 0 to 10 {
    // Referenced by %jj
    affine.for %arg2 = 0 to 20 {
    }
  }
}
```

For a more complicated example, we demonstrate 3-D tiling using krnl.block in
conjunction with krnl.permute:
```
%ii, %jj, %kk = krnl.define_loops 3
// Blocking each loop by a factor of 4.
%ib, %il = krnl.block %ii 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
%jb, %jl = krnl.block %jj 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
%kb, %kl = krnl.block %kk 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// Move iteration over tile coordinates to be the outer loops and iterateion over
// the inter-tile elements to be the inner loops.
krnl.permute(%ib, %il, %jb, %jl, %kb, %kl) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
krnl.iterate(%ib, %il, %jb, %jl, %kb, %kl) with (%ii -> %i = 0 to 1024, %jj -> %j = 0 to 2048, %kk -> %k = 0 to 4096)  {
}
```

The above IR gets lowered to:
```
affine.for %arg0 = 0 to 1024 step 4 {
  affine.for %arg1 = 0 to 2048 step 4 {
    affine.for %arg2 = 0 to 4096 step 4 {
      affine.for %arg3 = #map0(%arg0) to #map1(%arg0) {
        affine.for %arg4 = #map0(%arg1) to #map1(%arg1) {
          affine.for %arg5 = #map0(%arg2) to #map1(%arg2) {
          }
        }
      }
    }
  }
}
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`map` | ::mlir::ArrayAttr | 64-bit integer array attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`loops` | any type

### `krnl.shape` (::mlir::KrnlShapeOp)

Krnl operation to retreieve the shape of a MemRef.

Extracts the shape of a MemRef:
```
  "krnl.shape"(%memref)
```
The return result is of `shape.type`.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`alloc` | memref of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
`shape` | memref of any type values

### `krnl.specialized_kernel` (::mlir::KrnlSpecializedKernel)

Krnl specialized kernel op


Syntax:

```
operation ::= `krnl.specialized_kernel` `(` $loops `)` attr-dict `:` type($loops)
```

Krnl operation to convert.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`loops` | any type

### `krnl.store` (::mlir::KrnlStoreOp)

A Krnl operation to store data to the memref.


Syntax:

```
operation ::= `krnl.store` $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
```

The `krnl.store` stores a value to a memref location given by indices. The
value stored should have the same type as the elemental type of the memref.
The number of arguments provided within brackets need to match the rank of
the memref.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`value` | any type
`memref` | memref of any type values
`indices` | index

### `krnl.tan` (::mlir::KrnlTanOp)

Krnl tan scalar operation

Krnl tan scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
`out` | floating-point

### `krnl.terminate` (::mlir::KrnlTerminatorOp)

Krnl terminator operation

Krnl terminator is a special terminator operation for blocks inside krnl
iterate operations. It unconditionally transmits the control flow to the
successor of the operation enclosing the region.

This operation does _not_ have a custom syntax. However, krnl control
operations omit the terminator in their custom syntax for brevity.

### `krnl.unroll` (::mlir::KrnlUnrollOp)

Krnl unroll operation


Syntax:

```
operation ::= `krnl.unroll` $loop attr-dict `:` type($loop)
```

Fully unroll the specified loops.
```
krnl.unroll %i
```
unrolls the loop referred to by %i fully.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`loop` | any type

### `krnl.vector_type_cast` (::mlir::KrnlVectorTypeCastOp)

vector type cast operation


Syntax:

```
operation ::= `krnl.vector_type_cast` $source attr-dict `:` type($source) `to` type($result)
```

The "vector_type_cast" operation converts a memref from an non-vector
element type to another memref of a vector elemental type while not changing
the source memref's element type. The last dimension size of the source
dimension is divided (floor division) by the vector size to obtain the
corresponding dimension for target memref type.

%MV = vector_type_cast %M : memref<64x16xf32> to memref<64x2xvector<8xf32>>
%AV = vector_type_cast %A : memref<?x?xf32> to memref<?x?xvector<8xf32>>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`source` | memref of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
`result` | memref of any type values

