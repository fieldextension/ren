from enum import IntEnum, Enum, auto


class Op(IntEnum):
    # -- ALU --
    ADD = auto()
    SUB = auto()
    MUL = auto()
    IDIV = auto()  # Integer division
    NEG = auto()
    MOD_REDUCE = auto()  # Modular reduction: brings value back to canonical form [0, p) per RNS limb

    # -- Ring: high-level HE ops lowered during scheduling --
    AUTOMORPHISM = auto()  # Galois automorphism sigma_k: X -> X^k, permutes coefficients (arg=k)
    RESCALE = auto()  # Rescale a polynomial to the destination RNS basis
    ROTATE = auto()  # Multiply a polynomial by X^k (arg=k)
    TO_RING = auto()  # Ring conversion: extension (adds limbs) or view/subset (reinterprets with offset)

    # -- Conversion: domain/representation transforms that require dedicated kernels --
    TO_RNS = auto()  # Decompose single-modulus value into RNS (multi-limb) representation
    FROM_RNS = auto()  # Reconstruct single-modulus value from RNS limbs
    NTT = auto()  # Forward Number Theoretic Transform: COEFF -> NTT domain
    INTT = auto()  # Inverse NTT: NTT -> COEFF domain

    # -- Graph structure --
    SINK = auto()  # Root node collecting all outputs; src = tuple of result nodes
    CONST = auto()  # Scalar constant (arg=value); ring semantics applied by AS_RING wrapper

    # -- Elementwise indexing and lowered kernel ops --
    AND = auto()  # Bitwise AND (index masking, e.g. idx & (n-1) for coeff-index extraction)
    MOD = auto()  # Integer modulo
    BITREVERSE = auto()  # Reverse bottom arg bits of src (arg=n_power), used in NTT index mapping
    WHERE = auto()  # Ternary select: WHERE(cond, true_val, false_val)
    CMPEQ = auto()  # Equality comparison, returns bool
    CMPLT = auto()  # Less-than comparison, returns bool
    CAST = auto()  # Integer width conversion to dtype: widening zero-extends, narrowing keeps the low bits
    SHL = auto()  # Left shift by src[1] in [0, width), wrapping at the result width
    SHR = auto()  # Logical right shift by src[1] in [0, width)

    # -- Kernel IR: indexing + memory ops in lowered kernels --
    PARAM = auto()  # Positional buffer parameter in an opaque function or call body
    CONST_ARRAY = auto()  # Immutable typed values addressed by INDEX/LOAD
    GIDX = auto()  # Global thread index (arg=dim: 0=X, 1=Y; None for 1D)
    INDEX = auto()  # Pointer arithmetic: INDEX(slot, idx_expr) computes address
    LOAD = auto()  # Dereference: LOAD(INDEX(...)) reads from buffer
    STORE = auto()  # Write: STORE(INDEX(target, idx), value) writes value to buffer

    # -- Scheduling / materialization --
    BUFFERIZE = auto()  # Marks a materialization point - lowered to ALLOC + STORE + AFTER
    CONTIGUOUS = auto()  # Marks "needs contiguous access" - lowered to BUFFER_VIEW or BUFFERIZE
    BUFFER_VIEW = auto()  # Zero-copy view into a buffer: src=(base,), arg=(size, offset)
    COPY = auto()  # Device-to-device transfer: COPY(src, DEVICE(target_device))
    BUFFER = auto()  # Owned storage: arg=ParamArg with backing Buffer
    ALLOC = auto()  # Storage request bound per invocation: arg=ParamArg
    DEVICE = auto()  # Device tag (arg=device_name string, e.g. "cuda:0")
    AS_RING = auto()  # Metadata wrapper: attaches ring + domain without changing storage
    CALL = auto()  # Callable body and positional input/output bindings: src=(body, bindings...)
    LINEAR = auto()  # Ordered sequence of CALLs
    AFTER = auto()  # Dependency edge: AFTER(value, *deps) passes through value after deps complete

    # -- Randomness --
    RANDOM = auto()  # Generate a random ring element according to a specified distribution

    def __str__(self):
        return Enum.__str__(self)

    def __repr__(self):
        return str(self)

    def is_commutative(self) -> bool:
        return self in GroupOp.Commutative


class GroupOp:
    Unary = frozenset({Op.NEG, Op.MOD_REDUCE, Op.BITREVERSE})
    Binary = frozenset({Op.ADD, Op.SUB, Op.MUL, Op.IDIV, Op.AND, Op.MOD, Op.CMPEQ, Op.CMPLT, Op.SHL, Op.SHR})
    Ternary = frozenset({Op.WHERE})
    ALU = Unary | Binary | Ternary
    Elementwise = ALU | {Op.CAST}

    Ring = frozenset({Op.ROTATE, Op.AUTOMORPHISM})
    Conversion = frozenset({Op.NTT, Op.INTT, Op.AUTOMORPHISM, Op.TO_RNS, Op.FROM_RNS, Op.TO_RING, Op.RESCALE, Op.COPY})
    Memory = frozenset({Op.PARAM, Op.GIDX, Op.INDEX, Op.LOAD, Op.STORE})
    Defines = frozenset({Op.BUFFER, Op.ALLOC, Op.PARAM})
    Metadata = frozenset({Op.AS_RING, Op.AFTER})

    # Lowered: high-level ring ops that are rewritten to primitives during scheduling
    Lowered = frozenset({Op.ROTATE, Op.AUTOMORPHISM})
    # Op properties
    Commutative = frozenset({Op.ADD, Op.MUL, Op.AND, Op.CMPEQ})
    Associative = frozenset({Op.ADD, Op.MUL, Op.AND})

    # Ops that require reduced (mod-canonical) inputs
    ReducedInputs = frozenset({Op.SUB, Op.NEG, Op.MUL})

    # Ops that define storage or do not need a BUFFERIZE marker.
    AlwaysRealized = Defines | frozenset({Op.CONST, Op.STORE, Op.BUFFERIZE, Op.CONTIGUOUS, Op.DEVICE})
