--------- Generic Combinators ---------

def imap 'a : (n: i64) -> (i64 -> a) -> [n]a =
  \n f -> map f (iota n)

def imap1 = imap

def imap2 'a : (m: i64) -> (n: i64) -> (i64 -> i64 -> a) -> [m][n]a =
  \m n f -> imap m (\i -> imap n (f i))

def imap3 'a : (m: i64)
-> (n: i64)
-> (k: i64)
-> (i64 -> i64 -> i64 -> a) -> [m][n][k]a =
  \m n k f -> imap m (\i -> imap2 n k (f i))

def imap4 'a : (m: i64)
-> (n: i64)
-> (k: i64)
-> (l: i64)
-> (i64 -> i64 -> i64 -> i64 -> a) -> [m][n][k][l]a =
  \m n k l f -> imap m (\i -> imap3 n k l (f i))

def imap5 'a : (m: i64)
-> (n: i64)
-> (k: i64)
-> (l: i64)
-> (t: i64)
-> (i64 -> i64 -> i64 -> i64 -> i64 -> a) -> [m][n][k][l][t]a =
  \m n k l t f -> imap m (\i -> imap4 n k l t (f i))

def imap6 'a : (m: i64)
-> (n: i64)
-> (k: i64)
-> (l: i64)
-> (t: i64)
-> (p: i64)
-> (i64 -> i64 -> i64 -> i64 -> i64 -> i64 -> a) -> [m][n][k][l][t][p]a =
  \m n k l t p f -> imap m (\i -> imap5 n k l t p (f i))

def imap7 'a : (m: i64)
-> (n: i64)
-> (k: i64)
-> (l: i64)
-> (t: i64)
-> (p: i64)
-> (q: i64)
-> (i64 -> i64 -> i64 -> i64 -> i64 -> i64 -> i64 -> a) -> [m][n][k][l][t][p][q]a =
  \m n k l t p q f -> imap m (\i -> imap6 n k l t p q (f i))

def unzip7 [n] 'a 'b 'c 'd 'e 'f 'g : (a: [n](a, b, c, d, e, f, g)) -> ([n]a, [n]b, [n]c, [n]d, [n]e, [n]f, [n]g) =
  \a ->
    ( imap n (\i -> a[i].0)
    , imap n (\i -> a[i].1)
    , imap n (\i -> a[i].2)
    , imap n (\i -> a[i].3)
    , imap n (\i -> a[i].4)
    , imap n (\i -> a[i].5)
    , imap n (\i -> a[i].6)
    )

--==== MGPT Module ====--
module nn (F: real) = {
  type real = F.t

  def fromi64 (n: i64) = F.from_fraction n 1 -- why from fraction?
  def zero = fromi64 0
  def one = fromi64 1

  def isum1 : (m: i64) -> (i64 -> real) -> real =
    \m f -> loop r = zero for i < m do r F.+ f i

  def isum2 : (m: i64)
  -> (n: i64)
  -> (i64 -> i64 -> real) -> real =
    \m n f -> loop r = zero for i < m do r F.+ isum1 n (f i)

  def isum3 : (m: i64)
  -> (n: i64)
  -> (k: i64)
  -> (i64 -> i64 -> i64 -> real) -> real =
    \n m k f -> loop r = zero for i < n do r F.+ isum2 m k (f i)

  def isum4 : (m: i64)
  -> (n: i64)
  -> (k: i64)
  -> (l: i64)
  -> (i64 -> i64 -> i64 -> i64 -> real) -> real =
    \n m k l f -> loop r = zero for i < n do r F.+ isum3 m k l (f i)

  def isum5 : (m: i64)
  -> (n: i64)
  -> (k: i64)
  -> (l: i64)
  -> (t: i64)
  -> (i64 -> i64 -> i64 -> i64 -> i64 -> real) -> real =
    \n m k l t f -> loop r = zero for i < n do r F.+ isum4 m k l t (f i)

  def sum (a: []real) : real =
    reduce (F.+) zero a

  def imaximum1 : (m: i64) -> (i64 -> real) -> real =
    \m f -> F.maximum (imap1 m f)

  def imaximum2 : (m: i64)
  -> (n: i64)
  -> (i64 -> i64 -> real) -> real =
    \m n f -> F.maximum (imap1 m (\i -> imaximum1 n (f i)))

  def isoftmax1 (m: i64) (f : i64 -> real) : [m]real =
    -- #[noinline]
    let max = imaximum1 m f
    let exps = imap1 m (\x -> F.exp((f x) F.+ F.neg max))
    let scale = isum1 m (\x -> exps[x])
    in imap1 m (\x -> exps[x] F./ scale)

  --==== 2d cases ====--
  def sum2d (a: [][]real) : real =
    sum (map sum a)

  --==== Logistics ====--
  def logistics : real -> real =
    \e -> one F./ (one F.+ F.exp (F.neg e))

  def sgnp : real -> real = \e -> F.sgn (F.max e zero)

  def indicatorp : real -> real = sgnp

  --==== This is the generated function. ====--

  def forward_seq : (mask: [16][16]real)
    -> (wpe: [16][16]real)
    -> (wqry: [16][16]real)
    -> (wkey: [16][16]real)
    -> (wval: [16][16]real)
    -> (wout: [16][16]real)
    -> (wup: [64][16]real)
    -> (wdown: [16][64]real)
    -> (wvoc: [27][16]real)
    -> (wseq: [16][16]real)
    -- -> [16][27]real =
    -> [16][27]real =
    #[unsafe]
    \(mask: [16][16]real) (wpe: [16][16]real)
    (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real)
    (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real)
    (wvoc: [27][16]real) (wseq: [16][16]real) -> --(imap2 16 27 (\n m -> one F./ zero))

(let x0 = (imap2 16 16 (\x19_0 x19_1 -> (wpe[x19_0][x19_1] F.+ wseq[x19_0][x19_1])))
in (let x1 = (imap2 16 16 (\x20_0 x20_1 -> (let x21 = (imap1 16 (\x24_0 -> (x0[x20_0][x24_0] F.* x0[x20_0][x24_0])))
in (let x22 = ((isum1 16 (\x25_0 -> x21[x25_0])) F./ fromi64 16)
in (let x23 = (F.sqrt (x22 F.+ (one F./ fromi64 100000)))
in (x0[x20_0][x20_1] F.* (one F./ x23)))))))
in (let x2 = (imap2 16 16 (\x26_0 x26_1 -> (let x27 = (imap1 16 (\x30_0 -> (x1[x26_0][x30_0] F.* x1[x26_0][x30_0])))
in (let x28 = ((isum1 16 (\x31_0 -> x27[x31_0])) F./ fromi64 16)
in (let x29 = (F.sqrt (x28 F.+ (one F./ fromi64 100000)))
in (x1[x26_0][x26_1] F.* (one F./ x29)))))))
in (let x3 = (imap2 16 16 (\x32_0 x32_1 -> (isum1 16 (\x33_0 -> (wqry[x32_1][x33_0] F.* x2[x32_0][x33_0])))))
in (let x4 = (imap2 16 16 (\x34_0 x34_1 -> (isum1 16 (\x35_0 -> (wkey[x34_1][x35_0] F.* x2[x34_0][x35_0])))))
in (let x5 = (imap2 16 16 (\x36_0 x36_1 -> (isum1 16 (\x37_0 -> (wval[x36_1][x37_0] F.* x2[x36_0][x37_0])))))
in (let x6 = (imap3 4 16 4 (\x38_0 x38_1 x38_2 -> x3[x38_1][((x38_0 * 4) + x38_2)]))
in (let x7 = (imap3 4 16 4 (\x39_0 x39_1 x39_2 -> x4[x39_1][((x39_0 * 4) + x39_2)]))
in (let x8 = (imap3 4 16 4 (\x40_0 x40_1 x40_2 -> x5[x40_1][((x40_0 * 4) + x40_2)]))
in (let x9 = (imap3 4 16 4 (\x41_0 x41_1 x41_2 -> (let x42 = (imap2 16 16 (\x45_0 x45_1 -> (isum1 4 (\x46_0 -> (x6[x41_0][x45_0][x46_0] F.* x7[x41_0][x45_1][x46_0])))))
in (let x43 = (imap2 16 16 (\x47_0 x47_1 -> ((x42[x47_0][x47_1] F./ fromi64 2) F.+ mask[x47_0][x47_1])))
in (let x44 = (imap2 16 16 (\x48_0 x48_1 -> (let x49 = (imaximum1 16 (\x49_0 -> x43[x48_0][x49_0]))
in (let x50 = (imap1 16 (\x50_0 -> F.exp (x43[x48_0][x50_0] F.+ F.neg x49)))
in (let x51 = (isum1 16 (\x51_0 -> x50[x51_0]))
in (x50[x48_1] F./ x51))))))
in (isum1 16 (\x52_0 -> (x44[x41_1][x52_0] F.* x8[x41_0][x52_0][x41_2]))))))))
in (let x10 = (imap2 16 16 (\x53_0 x53_1 -> x9[(x53_1 / 4)][x53_0][(x53_1 % 4)]))
in (let x11 = (imap2 16 16 (\x54_0 x54_1 -> (isum1 16 (\x55_0 -> (wout[x54_1][x55_0] F.* x10[x54_0][x55_0])))))
in (let x12 = (imap2 16 16 (\x56_0 x56_1 -> (x11[x56_0][x56_1] F.+ x1[x56_0][x56_1])))
in (let x13 = (imap2 16 16 (\x57_0 x57_1 -> (let x58 = (imap1 16 (\x61_0 -> (x12[x57_0][x61_0] F.* x12[x57_0][x61_0])))
in (let x59 = ((isum1 16 (\x62_0 -> x58[x62_0])) F./ fromi64 16)
in (let x60 = (F.sqrt (x59 F.+ (one F./ fromi64 100000)))
in (x12[x57_0][x57_1] F.* (one F./ x60)))))))
in (let x14 = (imap2 16 64 (\x63_0 x63_1 -> (isum1 16 (\x64_0 -> (wup[x63_1][x64_0] F.* x13[x63_0][x64_0])))))
in (let x15 = (imap2 16 64 (\x65_0 x65_1 -> F.max x14[x65_0][x65_1] zero))
in (let x16 = (imap2 16 16 (\x66_0 x66_1 -> (isum1 64 (\x67_0 -> (wdown[x66_1][x67_0] F.* x15[x66_0][x67_0])))))
in (let x17 = (imap2 16 16 (\x68_0 x68_1 -> (x16[x68_0][x68_1] F.+ x12[x68_0][x68_1])))
in (imap2 16 27 (\x18_0 x18_1 -> (isum1 16 (\x69_0 -> (wvoc[x18_1][x69_0] F.* x17[x18_0][x69_0])))))))))))))))))))))))



--   def cal_loss : (mask: [16][16]real)
--     -> (wpe: [16][16]real)
--     -> (wqry: [16][16]real)
--     -> (wkey: [16][16]real)
--     -> (wval: [16][16]real)
--     -> (wout: [16][16]real)
--     -> (wup: [64][16]real)
--     -> (wdown: [16][64]real)
--     -> (wvoc: [27][16]real)
--     -> (wseq: [16][16]real)
--     -> (target: [16][27]real)
--     -> (real, [16]real) =
--     #[unsafe]
--     \(mask: [16][16]real) (wpe: [16][16]real)
--     (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real)
--     (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real)
--     (wvoc: [27][16]real) (wseq: [16][16]real) (target: [16][27]real) ->

  def grad_loss : (mask: [16][16]real)
    -> (wpe: [16][16]real)
    -> (wqry: [16][16]real)
    -> (wkey: [16][16]real)
    -> (wval: [16][16]real)
    -> (wout: [16][16]real)
    -> (wup: [64][16]real)
    -> (wdown: [16][64]real)
    -> (wvoc: [27][16]real)
    -> (wseq: [16][16]real)
    -> (target: [16][27]real)
    -> ([16][16]real, -- dwpe
        [16][16]real, -- dwqry
        [16][16]real, -- dwkey
        [16][16]real, -- dwval
        [16][16]real, -- dwout
        [64][16]real, -- dwup
        [16][64]real, -- dwdown
        [27][16]real, -- dwvoc
        [16][16]real -- dwseq
        ) =
    #[unsafe]
    \(mask: [16][16]real) (wpe: [16][16]real)
    (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real)
    (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real)
    (wvoc: [27][16]real) (wseq: [16][16]real) (target: [16][27]real) ->
    -- (wpe, wqry, wkey, wval, wout, wup, wdown, wvoc, wseq)

let x0 = (imap2 16 16 (\x1_0 x1_1 -> (wpe[x1_0][x1_1] F.+ wseq[x1_0][x1_1])))
let x2 = (imap2 16 16 (\x3_0 x3_1 -> (let x4 = ((isum1 16 (\x6_0 -> (x0[x3_0][x6_0] F.* x0[x3_0][x6_0]))) F./ fromi64 16)
in (let x5 = (F.sqrt (x4 F.+ (one F./ fromi64 100000)))
in (x0[x3_0][x3_1] F.* (one F./ x5))))))
let x7 = (imap2 16 16 (\x8_0 x8_1 -> (let x9 = ((isum1 16 (\x11_0 -> (x2[x8_0][x11_0] F.* x2[x8_0][x11_0]))) F./ fromi64 16)
in (let x10 = (F.sqrt (x9 F.+ (one F./ fromi64 100000)))
in (x2[x8_0][x8_1] F.* (one F./ x10))))))
let x12 = (imap2 16 16 (\x13_0 x13_1 -> (isum1 16 (\x14_0 -> (wqry[x13_1][x14_0] F.* x7[x13_0][x14_0])))))
let x15 = (imap2 16 16 (\x16_0 x16_1 -> (isum1 16 (\x17_0 -> (wkey[x16_1][x17_0] F.* x7[x16_0][x17_0])))))
let x18 = (imap2 16 16 (\x19_0 x19_1 -> (isum1 16 (\x20_0 -> (wval[x19_1][x20_0] F.* x7[x19_0][x20_0])))))
let x21 = (imap3 4 16 4 (\x22_0 x22_1 x22_2 -> x12[x22_1][((x22_0 * 4) + x22_2)]))
let x23 = (imap3 4 16 4 (\x24_0 x24_1 x24_2 -> x15[x24_1][((x24_0 * 4) + x24_2)]))
let x25 = (imap3 4 16 4 (\x26_0 x26_1 x26_2 -> x18[x26_1][((x26_0 * 4) + x26_2)]))
let x27 = (imap3 4 16 4 (\x28_0 x28_1 x28_2 -> (let x29 = (imap2 16 16 (\x32_0 x32_1 -> (isum1 4 (\x33_0 -> (x21[x28_0][x32_0][x33_0] F.* x23[x28_0][x32_1][x33_0])))))
in (let x30 = (imap2 16 16 (\x34_0 x34_1 -> ((x29[x34_0][x34_1] F./ fromi64 2) F.+ mask[x34_0][x34_1])))
in (let x31 = (imap2 16 16 (\x35_0 x35_1 -> (let x36 = (isoftmax1 16 (\x36_0 -> x30[x35_0][x36_0]))
in x36[x35_1])))
in (isum1 16 (\x37_0 -> (x31[x28_1][x37_0] F.* x25[x28_0][x37_0][x28_2]))))))))
let x38 = (imap2 16 16 (\x39_0 x39_1 -> x27[(x39_1 / 4)][x39_0][(x39_1 % 4)]))
let x40 = (imap2 16 16 (\x41_0 x41_1 -> (isum1 16 (\x42_0 -> (wout[x41_1][x42_0] F.* x38[x41_0][x42_0])))))
let x43 = (imap2 16 16 (\x44_0 x44_1 -> (x40[x44_0][x44_1] F.+ x2[x44_0][x44_1])))
let x45 = (imap2 16 16 (\x46_0 x46_1 -> (let x47 = ((isum1 16 (\x49_0 -> (x43[x46_0][x49_0] F.* x43[x46_0][x49_0]))) F./ fromi64 16)
in (let x48 = (F.sqrt (x47 F.+ (one F./ fromi64 100000)))
in (x43[x46_0][x46_1] F.* (one F./ x48))))))
let x50 = (imap2 16 64 (\x51_0 x51_1 -> (isum1 16 (\x52_0 -> (wup[x51_1][x52_0] F.* x45[x51_0][x52_0])))))
let x53 = (imap2 16 64 (\x54_0 x54_1 -> F.max x50[x54_0][x54_1] zero))
let x55 = (imap2 16 16 (\x56_0 x56_1 -> (isum1 64 (\x57_0 -> (wdown[x56_1][x57_0] F.* x53[x56_0][x57_0])))))
let x58 = (imap2 16 16 (\x59_0 x59_1 -> (x55[x59_0][x59_1] F.+ x43[x59_0][x59_1])))
let x60 = (imap2 16 27 (\x61_0 x61_1 -> (isum1 16 (\x62_0 -> (wvoc[x61_1][x62_0] F.* x58[x61_0][x62_0])))))
let x63 = (imap1 16 (\x64_0 -> (one F./ fromi64 16)))
let x65 = (imap2 16 27 (\x66_0 x66_1 -> (let x67 = (isoftmax1 27 (\x67_0 -> x60[x66_0][x67_0]))
in (F.log x67[x66_1]))))
let x68 = (imap2 16 27 (\x69_0 x69_1 -> ((F.neg x63[x69_0]) F.* target[x69_0][x69_1])))
let x70 = (imap2 16 27 (\x71_0 x71_1 -> (let x72 = (isoftmax1 27 (\x72_0 -> x60[x71_0][x72_0]))
in x72[x71_1])))
let x73 = (imap2 16 27 (\x74_0 x74_1 -> (let x75 = (isoftmax1 27 (\x75_0 -> x60[x74_0][x75_0]))
in (x68[x74_0][x74_1] F.* (one F./ x75[x74_1])))))
let x76 = (imap1 16 (\x77_0 -> (isum1 27 (\x78_0 -> (x73[x77_0][x78_0] F.* x70[x77_0][x78_0])))))
let x79 = (imap2 16 27 (\x80_0 x80_1 -> (x70[x80_0][x80_1] F.* (x73[x80_0][x80_1] F.+ (F.neg x76[x80_0])))))
let x81 = (imap2 16 16 (\x82_0 x82_1 -> (isum1 27 (\x83_0 -> (wvoc[x83_0][x82_1] F.* x79[x82_0][x83_0])))))
let x84 = (imap2 16 64 (\x85_0 x85_1 -> (isum1 16 (\x86_0 -> (wdown[x86_0][x85_1] F.* x81[x85_0][x86_0])))))
let x87 = (imap2 16 64 (\x88_0 x88_1 -> ((indicatorp x50[x88_0][x88_1]) F.* x84[x88_0][x88_1])))
let x89 = (imap2 16 16 (\x90_0 x90_1 -> (isum1 64 (\x91_0 -> (wup[x91_0][x90_1] F.* x87[x90_0][x91_0])))))
let x92 = (imap2 16 16 (\x93_0 x93_1 -> (x43[x93_0][x93_1] F.* x43[x93_0][x93_1])))
let x94 = (imap1 16 (\x95_0 -> ((isum1 16 (\x96_0 -> x92[x95_0][x96_0])) F./ fromi64 16)))
let x97 = (imap1 16 (\x98_0 -> (F.sqrt (x94[x98_0] F.+ (one F./ fromi64 100000)))))
let x99 = (imap1 16 (\x100_0 -> (F.neg ((one F./ x97[x100_0]) F.* ((isum1 16 (\x101_0 -> (x43[x100_0][x101_0] F.* x89[x100_0][x101_0]))) F.* (one F./ x97[x100_0]))))))
let x102 = (imap1 16 (\x103_0 -> (x99[x103_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x94[x103_0] F.+ (one F./ fromi64 100000))))))))
let x104 = (imap2 16 16 (\x105_0 x105_1 -> (x102[x105_0] F./ fromi64 16)))
let x106 = (imap2 16 16 (\x107_0 x107_1 -> (x81[x107_0][x107_1] F.+ (((x89[x107_0][x107_1] F.* (one F./ x97[x107_0])) F.+ (x43[x107_0][x107_1] F.* x104[x107_0][x107_1])) F.+ (x104[x107_0][x107_1] F.* x43[x107_0][x107_1])))))
let x108 = (imap2 16 16 (\x109_0 x109_1 -> (isum1 16 (\x110_0 -> (wout[x110_0][x109_1] F.* x106[x109_0][x110_0])))))
let x111 = (imap3 4 16 4 (\x112_0 x112_1 x112_2 -> x108[x112_1][((x112_0 * 4) + x112_2)]))
let x113 = (imap3 4 16 16 (\x114_0 x114_1 x114_2 -> (isum1 4 (\x115_0 -> (x21[x114_0][x114_1][x115_0] F.* x23[x114_0][x114_2][x115_0])))))
let x116 = (imap3 4 16 16 (\x117_0 x117_1 x117_2 -> ((x113[x117_0][x117_1][x117_2] F./ fromi64 2) F.+ mask[x117_1][x117_2])))
let x118 = (imap3 4 16 16 (\x119_0 x119_1 x119_2 -> (let x120 = (isoftmax1 16 (\x120_0 -> x116[x119_0][x119_1][x120_0]))
in x120[x119_2])))
let x121 = (imap3 4 16 16 (\x122_0 x122_1 x122_2 -> (isum1 4 (\x123_0 -> (x111[x122_0][x122_1][x123_0] F.* x25[x122_0][x122_2][x123_0])))))
let x124 = (imap3 4 16 16 (\x125_0 x125_1 x125_2 -> x121[x125_0][x125_1][x125_2]))
let x126 = (imap2 4 16 (\x127_0 x127_1 -> (isum1 16 (\x128_0 -> (x124[x127_0][x127_1][x128_0] F.* x118[x127_0][x127_1][x128_0])))))
let x129 = (imap3 4 16 16 (\x130_0 x130_1 x130_2 -> (x118[x130_0][x130_1][x130_2] F.* (x124[x130_0][x130_1][x130_2] F.+ (F.neg x126[x130_0][x130_1])))))
let x131 = (imap3 4 16 16 (\x132_0 x132_1 x132_2 -> (x129[x132_0][x132_1][x132_2] F./ fromi64 2)))
let x133 = (imap3 4 16 4 (\x134_0 x134_1 x134_2 -> (isum1 16 (\x135_0 -> (x118[x134_0][x135_0][x134_1] F.* x111[x134_0][x135_0][x134_2])))))
let x136 = (imap3 4 16 4 (\x137_0 x137_1 x137_2 -> (isum1 16 (\x138_0 -> (x21[x137_0][x138_0][x137_2] F.* x131[x137_0][x138_0][x137_1])))))
let x139 = (imap3 4 16 4 (\x140_0 x140_1 x140_2 -> (isum1 16 (\x141_0 -> (x131[x140_0][x140_1][x141_0] F.* x23[x140_0][x141_0][x140_2])))))
let x142 = (imap2 16 16 (\x143_0 x143_1 -> x133[(x143_1 / 4)][x143_0][(x143_1 % 4)]))
let x144 = (imap2 16 16 (\x145_0 x145_1 -> x136[(x145_1 / 4)][x145_0][(x145_1 % 4)]))
let x146 = (imap2 16 16 (\x147_0 x147_1 -> x139[(x147_1 / 4)][x147_0][(x147_1 % 4)]))
let x148 = (imap2 16 16 (\x149_0 x149_1 -> (((isum1 16 (\x150_0 -> (wval[x150_0][x149_1] F.* x142[x149_0][x150_0]))) F.+ (isum1 16 (\x151_0 -> (wkey[x151_0][x149_1] F.* x144[x149_0][x151_0])))) F.+ (isum1 16 (\x152_0 -> (wqry[x152_0][x149_1] F.* x146[x149_0][x152_0]))))))
let x153 = (imap2 16 16 (\x154_0 x154_1 -> (x2[x154_0][x154_1] F.* x2[x154_0][x154_1])))
let x155 = (imap1 16 (\x156_0 -> ((isum1 16 (\x157_0 -> x153[x156_0][x157_0])) F./ fromi64 16)))
let x158 = (imap1 16 (\x159_0 -> (F.sqrt (x155[x159_0] F.+ (one F./ fromi64 100000)))))
let x160 = (imap1 16 (\x161_0 -> (F.neg ((one F./ x158[x161_0]) F.* ((isum1 16 (\x162_0 -> (x2[x161_0][x162_0] F.* x148[x161_0][x162_0]))) F.* (one F./ x158[x161_0]))))))
let x163 = (imap1 16 (\x164_0 -> (x160[x164_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x155[x164_0] F.+ (one F./ fromi64 100000))))))))
let x165 = (imap2 16 16 (\x166_0 x166_1 -> (x163[x166_0] F./ fromi64 16)))
let x167 = (imap2 16 16 (\x168_0 x168_1 -> (x106[x168_0][x168_1] F.+ (((x148[x168_0][x168_1] F.* (one F./ x158[x168_0])) F.+ (x2[x168_0][x168_1] F.* x165[x168_0][x168_1])) F.+ (x165[x168_0][x168_1] F.* x2[x168_0][x168_1])))))
let x169 = (imap2 16 16 (\x170_0 x170_1 -> (x0[x170_0][x170_1] F.* x0[x170_0][x170_1])))
let x171 = (imap1 16 (\x172_0 -> ((isum1 16 (\x173_0 -> x169[x172_0][x173_0])) F./ fromi64 16)))
let x174 = (imap1 16 (\x175_0 -> (F.sqrt (x171[x175_0] F.+ (one F./ fromi64 100000)))))
let x176 = (imap1 16 (\x177_0 -> (F.neg ((one F./ x174[x177_0]) F.* ((isum1 16 (\x178_0 -> (x0[x177_0][x178_0] F.* x167[x177_0][x178_0]))) F.* (one F./ x174[x177_0]))))))
let x179 = (imap1 16 (\x180_0 -> (x176[x180_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x171[x180_0] F.+ (one F./ fromi64 100000))))))))
let x181 = (imap2 16 16 (\x182_0 x182_1 -> (x179[x182_0] F./ fromi64 16)))
let x183 = (imap2 16 16 (\x184_0 x184_1 -> (((x167[x184_0][x184_1] F.* (one F./ x174[x184_0])) F.+ (x0[x184_0][x184_1] F.* x181[x184_0][x184_1])) F.+ (x181[x184_0][x184_1] F.* x0[x184_0][x184_1]))))

let dmask = (imap2 16 16 (\x186_0 x186_1 -> (isum1 4 (\x185_0 -> x129[x185_0][x186_0][x186_1]))))
let dwpe = (imap2 16 16 (\x187_0 x187_1 -> x183[x187_0][x187_1]))
let dwqry = (imap2 16 16 (\x188_0 x188_1 -> (isum1 16 (\x189_0 -> (x146[x189_0][x188_0] F.* x7[x189_0][x188_1])))))
let dwkey = (imap2 16 16 (\x190_0 x190_1 -> (isum1 16 (\x191_0 -> (x144[x191_0][x190_0] F.* x7[x191_0][x190_1])))))
let dwval = (imap2 16 16 (\x192_0 x192_1 -> (isum1 16 (\x193_0 -> (x142[x193_0][x192_0] F.* x7[x193_0][x192_1])))))
let dwout = (imap2 16 16 (\x194_0 x194_1 -> (isum1 16 (\x195_0 -> (x106[x195_0][x194_0] F.* x38[x195_0][x194_1])))))
let dwup = (imap2 64 16 (\x196_0 x196_1 -> (isum1 16 (\x197_0 -> (x87[x197_0][x196_0] F.* x45[x197_0][x196_1])))))
let dwdown = (imap2 16 64 (\x198_0 x198_1 -> (isum1 16 (\x199_0 -> (x81[x199_0][x198_0] F.* x53[x199_0][x198_1])))))
let dwvoc = (imap2 27 16 (\x200_0 x200_1 -> (isum1 16 (\x201_0 -> (x79[x201_0][x200_0] F.* x58[x201_0][x200_1])))))
let dwseq = (imap2 16 16 (\x202_0 x202_1 -> x183[x202_0][x202_1]))
let dtarget = (imap2 16 27 (\x203_0 x203_1 -> (F.neg (x65[x203_0][x203_1] F.* x63[x203_0]))))

in (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq)
}


module nn64 = nn f64

type params = {
  wte:   [27][16]f64, -- token embeddings
  wpe:   [16][16]f64, -- position embeddings
  wqry:  [16][16]f64, -- query weights
  wkey:  [16][16]f64, -- key weights
  wval:  [16][16]f64, -- value weights
  wout:  [16][16]f64, -- output weights
  wup:   [64][16]f64, -- MLP up-projection
  wdown: [16][64]f64, -- MLP down-projection
  wvoc:  [27][16]f64  -- output projection
}

entry to_params (wte: [27][16]f64)  (wpe: [16][16]f64)
    (wqry: [16][16]f64) (wkey: [16][16]f64) (wval: [16][16]f64)
    (wout: [16][16]f64) (wup: [64][16]f64) (wdown: [16][64]f64)
    (wvoc: [27][16]f64) : params =
    {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc}

def from_params (p : params) :
  (
  [27][16]f64, -- dwte
  [16][16]f64, -- dwpe
  [16][16]f64, -- dwqry
  [16][16]f64, -- dwkey
  [16][16]f64, -- dwval
  [16][16]f64, -- dwout
  [64][16]f64, -- dwup
  [16][64]f64, -- dwdown
  [27][16]f64, -- dwvoc
  ) =
  let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
  in (wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc)

entry forward_seq (p : params) (tokens : [16]i64) (mask : [16][16]f64) : [16][27]f64 =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let wseq = (imap2 16 16 (\m n -> wte[tokens[m]][n]))
   in nn64.forward_seq mask wpe wqry wkey wval wout wup wdown wvoc wseq

-- entry cal_loss (p : params) (tokens : [16]i64) (target : [16][27]f64) (mask : [16][16]f64) : (f64 , [16]f64) =
--    let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
--    let wseq = (imap2 16 16 (\m n -> wte[tokens[m]][n]))
--    in nn64.cal_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target

def cal_target (n : i64) (tokens : [16]i64) : [16][27]f64 =
  imap2 16 27 (\i j -> (if ((i < (n - 1)) && (tokens[i + 1] == j)) then 1 else 0))

def adam_opt_w [n] [m] (w : [n][m]f64) (mw : [n][m]f64) (vw : [n][m]f64)
  (dw : [n][m]f64) (step : i64) (lt_r : f64):
  ([n][m]f64, [n][m]f64, [n][m]f64) =
  let new_mw = imap2 n m (\i j ->
    0.85 * mw[i][j] + ((1 - 0.85) * dw[i][j]))
  let new_vw = imap2 n m (\i j ->
    0.99 * vw[i][j] + ((1 - 0.99) * dw[i][j] * dw[i][j]))
  let m_hat = imap2 n m (\i j ->
    new_mw[i][j] / (1 - 0.85 ** ((nn64.fromi64 step) + 1)))
  let v_hat = imap2 n m (\i j ->
    new_vw[i][j] / (1 - (0.99 ** ((nn64.fromi64 step) + 1))))
  let new_w = imap2 n m (\i j ->
    w[i][j] - (lt_r * m_hat[i][j] / ((v_hat[i][j] ** 0.5) + 0.00000001)))
  in (new_w, new_mw, new_vw)

def adam_opt (p : params) (mp : params) (vp : params)
  (dp : params) (step : i64):
  (params,  params,  params) =
  let lt_r = 0.01 * (1 - (nn64.fromi64 step) / (nn64.fromi64 500))
  let (wte, mwte, vwte) =
    adam_opt_w p.wte mp.wte vp.wte dp.wte step lt_r
  let (wpe, mwpe, vwpe) =
    adam_opt_w p.wpe mp.wpe vp.wpe dp.wpe step lt_r
  let (wqry, mwqry, vwqry) =
    adam_opt_w p.wqry mp.wqry vp.wqry dp.wqry step lt_r
  let (wkey, mwkey, vwkey) =
    adam_opt_w p.wkey mp.wkey vp.wkey dp.wkey step lt_r
  let (wval, mwval, vwval) =
    adam_opt_w p.wval mp.wval vp.wval dp.wval step lt_r
  let (wout, mwout, vwout) =
    adam_opt_w p.wout mp.wout vp.wout dp.wout step lt_r
  let (wup, mwup, vwup) =
    adam_opt_w p.wup mp.wup vp.wup dp.wup step lt_r
  let (wdown, mwdown, vwdown) =
    adam_opt_w p.wdown mp.wdown vp.wdown dp.wdown step lt_r
  let (wvoc, mwvoc, vwvoc) =
    adam_opt_w p.wvoc mp.wvoc vp.wvoc dp.wvoc step lt_r
  let p' = to_params wte wpe wqry wkey wval wout wup wdown wvoc
  let mp' = to_params mwte mwpe mwqry mwkey mwval mwout mwup mwdown mwvoc
  let vp' = to_params vwte vwpe vwqry vwkey vwval vwout vwup vwdown vwvoc
  in (p', mp', vp')

def grad_loss (dl : i64) (p : params) (tokens : [16]i64) (mask : [16][16]f64) :
        (
        [27][16]f64, -- dwte
        [16][16]f64, -- dwpe
        [16][16]f64, -- dwqry
        [16][16]f64, -- dwkey
        [16][16]f64, -- dwval
        [16][16]f64, -- dwout
        [64][16]f64, -- dwup
        [16][64]f64, -- dwdown
        [27][16]f64, -- dwvoc
        ) =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   -- cal targets
   let targets = cal_target dl tokens
   -- cal voc embedding
   let wseq = (imap2 16 16 (\m n -> wte[tokens[m]][n]))
   -- cal gradient
   let (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq) =
    nn64.grad_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq targets
   let dwte = (imap2 27 16 (\m n -> nn64.isum1 16 (\k -> if (tokens[k] == m) then dwseq[k][n] else nn64.zero)))
   in  (dwte, dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc)


def cal_step (dl : i64) (p : params) (mp : params) (vp : params)
  (tokens : [16]i64) (mask : [16][16]f64)
  (step : i64) :
  (params,  params,  params) =
  -- cal gradient
  let (dwte, dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc) =
    grad_loss dl p tokens mask
  let dp = to_params dwte dwpe dwqry dwkey dwval dwout dwup dwdown dwvoc
  -- cal new model weights
  let (p', mp', vp') =
    adam_opt p mp vp dp step
  in (p', mp', vp')

entry train (p : params) (mp : params) (vp : params)
  (masks : [500][16][16]f64) (dls : [500]i64)
  (seqs : [500][16]i64) =
  let (new_p, new_mp, new_vp) =
    loop (p', mp', vp') = (p, mp, vp)
    for step < 500 do
      let dl = dls[step]
      let tokens = seqs[step]
      let mask = masks[step]
      in (cal_step dl p' mp' vp' tokens mask step)
  in ((from_params new_p), (from_params new_mp), (from_params new_vp))

entry zero_params : params =
  let wte = imap2 27 16 (\_ _ -> 0)
  let wpe = imap2 16 16 (\_ _ -> 0)
  let wqry = imap2 16 16 (\_ _ -> 0)
  let wkey = imap2 16 16 (\_ _ -> 0)
  let wval = imap2 16 16 (\_ _ -> 0)
  let wout = imap2 16 16 (\_ _ -> 0)
  let wup = imap2 64 16 (\_ _ -> 0)
  let wdown = imap2 16 64 (\_ _ -> 0)
  let wvoc = imap2 27 16 (\_ _ -> 0)
  in {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc}