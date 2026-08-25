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

  --==== 2d cases ====--
  def sum2d (a: [][]real) : real =
    sum (map sum a)

  --==== Logistics ====--
  def logistics : real -> real =
    \e -> one F./ (one F.+ F.exp (F.neg e))

  def sgnp : real -> real = \e -> F.sgn (F.max e zero)

  def indicatorp : real -> real = sgnp

  def imaximum1 : (m: i64) -> (i64 -> real) -> real =
    \m f -> F.maximum (imap1 m f)

  def imaximum2 : (m: i64)
  -> (n: i64)
  -> (i64 -> i64 -> real) -> real =
    \m n f -> F.maximum (imap1 m (\i -> imaximum1 n (f i)))

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
in (let x31 = (imap2 16 16 (\x35_0 x35_1 -> (let x36 = (imaximum1 16 (\x36_0 -> x30[x35_0][x36_0]))
in (let x37 = (imap1 16 (\x37_0 -> F.exp (x30[x35_0][x37_0] F.+ F.neg x36)))
in (let x38 = (isum1 16 (\x38_0 -> x37[x38_0]))
in (x37[x35_1] F./ x38))))))
in (isum1 16 (\x39_0 -> (x31[x28_1][x39_0] F.* x25[x28_0][x39_0][x28_2]))))))))
let x40 = (imap2 16 16 (\x41_0 x41_1 -> x27[(x41_1 / 4)][x41_0][(x41_1 % 4)]))
let x42 = (imap2 16 16 (\x43_0 x43_1 -> (isum1 16 (\x44_0 -> (wout[x43_1][x44_0] F.* x40[x43_0][x44_0])))))
let x45 = (imap2 16 16 (\x46_0 x46_1 -> (x42[x46_0][x46_1] F.+ x2[x46_0][x46_1])))
let x47 = (imap2 16 16 (\x48_0 x48_1 -> (let x49 = ((isum1 16 (\x51_0 -> (x45[x48_0][x51_0] F.* x45[x48_0][x51_0]))) F./ fromi64 16)
in (let x50 = (F.sqrt (x49 F.+ (one F./ fromi64 100000)))
in (x45[x48_0][x48_1] F.* (one F./ x50))))))
let x52 = (imap2 16 64 (\x53_0 x53_1 -> (isum1 16 (\x54_0 -> (wup[x53_1][x54_0] F.* x47[x53_0][x54_0])))))
let x55 = (imap2 16 64 (\x56_0 x56_1 -> F.max x52[x56_0][x56_1] zero))
let x57 = (imap2 16 16 (\x58_0 x58_1 -> (isum1 64 (\x59_0 -> (wdown[x58_1][x59_0] F.* x55[x58_0][x59_0])))))
let x60 = (imap2 16 16 (\x61_0 x61_1 -> (x57[x61_0][x61_1] F.+ x45[x61_0][x61_1])))
let x62 = (imap2 16 27 (\x63_0 x63_1 -> (isum1 16 (\x64_0 -> (wvoc[x63_1][x64_0] F.* x60[x63_0][x64_0])))))
let x65 = (imap1 16 (\x66_0 -> (one F./ fromi64 16)))
let x67 = (imap2 16 27 (\x68_0 x68_1 -> (let x69 = (imaximum1 27 (\x69_0 -> x62[x68_0][x69_0]))
in (let x70 = (imap1 27 (\x70_0 -> F.exp (x62[x68_0][x70_0] F.+ F.neg x69)))
in (let x71 = (isum1 27 (\x71_0 -> x70[x71_0]))
in (F.log (x70[x68_1] F./ x71)))))))
let x72 = (imap2 16 27 (\x73_0 x73_1 -> ((F.neg x65[x73_0]) F.* target[x73_0][x73_1])))
let x74 = (imap1 16 (\x75_0 -> (isum1 27 (\x76_0 -> (let x77 = (imaximum1 27 (\x77_0 -> x62[x75_0][x77_0]))
in (let x78 = (imap1 27 (\x78_0 -> F.exp (x62[x75_0][x78_0] F.+ F.neg x77)))
in (let x79 = (isum1 27 (\x79_0 -> x78[x79_0]))
in (let x80 = (imaximum1 27 (\x80_0 -> x62[x75_0][x80_0]))
in (let x81 = (imap1 27 (\x81_0 -> F.exp (x62[x75_0][x81_0] F.+ F.neg x80)))
in (let x82 = (isum1 27 (\x82_0 -> x81[x82_0]))
in ((x72[x75_0][x76_0] F.* (one F./ (x78[x76_0] F./ x79))) F.* (x81[x76_0] F./ x82))))))))))))
let x83 = (imap2 16 27 (\x84_0 x84_1 -> (let x85 = (imaximum1 27 (\x85_0 -> x62[x84_0][x85_0]))
in (let x86 = (imap1 27 (\x86_0 -> F.exp (x62[x84_0][x86_0] F.+ F.neg x85)))
in (let x87 = (isum1 27 (\x87_0 -> x86[x87_0]))
in (let x88 = (imaximum1 27 (\x88_0 -> x62[x84_0][x88_0]))
in (let x89 = (imap1 27 (\x89_0 -> F.exp (x62[x84_0][x89_0] F.+ F.neg x88)))
in (let x90 = (isum1 27 (\x90_0 -> x89[x90_0]))
in ((x86[x84_1] F./ x87) F.* ((x72[x84_0][x84_1] F.* (one F./ (x89[x84_1] F./ x90))) F.+ (F.neg x74[x84_0])))))))))))
let x91 = (imap2 16 16 (\x92_0 x92_1 -> (isum1 27 (\x93_0 -> (wvoc[x93_0][x92_1] F.* x83[x92_0][x93_0])))))
let x94 = (imap2 16 64 (\x95_0 x95_1 -> (isum1 16 (\x96_0 -> (wdown[x96_0][x95_1] F.* x91[x95_0][x96_0])))))
let x97 = (imap2 16 64 (\x98_0 x98_1 -> ((indicatorp x52[x98_0][x98_1]) F.* x94[x98_0][x98_1])))
let x99 = (imap2 16 16 (\x100_0 x100_1 -> (isum1 64 (\x101_0 -> (wup[x101_0][x100_1] F.* x97[x100_0][x101_0])))))
let x102 = (imap2 16 16 (\x103_0 x103_1 -> (x45[x103_0][x103_1] F.* x45[x103_0][x103_1])))
let x104 = (imap1 16 (\x105_0 -> ((isum1 16 (\x106_0 -> x102[x105_0][x106_0])) F./ fromi64 16)))
let x107 = (imap1 16 (\x108_0 -> (F.sqrt (x104[x108_0] F.+ (one F./ fromi64 100000)))))
let x109 = (imap1 16 (\x110_0 -> (F.neg ((one F./ x107[x110_0]) F.* ((isum1 16 (\x111_0 -> (x45[x110_0][x111_0] F.* x99[x110_0][x111_0]))) F.* (one F./ x107[x110_0]))))))
let x112 = (imap1 16 (\x113_0 -> (x109[x113_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x104[x113_0] F.+ (one F./ fromi64 100000))))))))
let x114 = (imap2 16 16 (\x115_0 x115_1 -> (x112[x115_0] F./ fromi64 16)))
let x116 = (imap2 16 16 (\x117_0 x117_1 -> (x91[x117_0][x117_1] F.+ (((x99[x117_0][x117_1] F.* (one F./ x107[x117_0])) F.+ (x45[x117_0][x117_1] F.* x114[x117_0][x117_1])) F.+ (x114[x117_0][x117_1] F.* x45[x117_0][x117_1])))))
let x118 = (imap2 16 16 (\x119_0 x119_1 -> (isum1 16 (\x120_0 -> (wout[x120_0][x119_1] F.* x116[x119_0][x120_0])))))
let x121 = (imap3 4 16 4 (\x122_0 x122_1 x122_2 -> x118[x122_1][((x122_0 * 4) + x122_2)]))
let x123 = (imap3 4 16 16 (\x124_0 x124_1 x124_2 -> (isum1 4 (\x125_0 -> (x21[x124_0][x124_1][x125_0] F.* x23[x124_0][x124_2][x125_0])))))
let x126 = (imap3 4 16 16 (\x127_0 x127_1 x127_2 -> ((x123[x127_0][x127_1][x127_2] F./ fromi64 2) F.+ mask[x127_1][x127_2])))
let x128 = (imap3 4 16 16 (\x129_0 x129_1 x129_2 -> (let x130 = (imaximum1 16 (\x130_0 -> x126[x129_0][x129_1][x130_0]))
in (let x131 = (imap1 16 (\x131_0 -> F.exp (x126[x129_0][x129_1][x131_0] F.+ F.neg x130)))
in (let x132 = (isum1 16 (\x132_0 -> x131[x132_0]))
in (x131[x129_2] F./ x132))))))
let x133 = (imap3 4 16 16 (\x134_0 x134_1 x134_2 -> (isum1 4 (\x135_0 -> (x121[x134_0][x134_1][x135_0] F.* x25[x134_0][x134_2][x135_0])))))
let x136 = (imap2 4 16 (\x137_0 x137_1 -> (isum1 16 (\x138_0 -> (let x139 = (imaximum1 16 (\x139_0 -> x126[x137_0][x137_1][x139_0]))
in (let x140 = (imap1 16 (\x140_0 -> F.exp (x126[x137_0][x137_1][x140_0] F.+ F.neg x139)))
in (let x141 = (isum1 16 (\x141_0 -> x140[x141_0]))
in (x133[x137_0][x137_1][x138_0] F.* (x140[x138_0] F./ x141)))))))))
let x142 = (imap3 4 16 16 (\x143_0 x143_1 x143_2 -> (let x144 = (imaximum1 16 (\x144_0 -> x126[x143_0][x143_1][x144_0]))
in (let x145 = (imap1 16 (\x145_0 -> F.exp (x126[x143_0][x143_1][x145_0] F.+ F.neg x144)))
in (let x146 = (isum1 16 (\x146_0 -> x145[x146_0]))
in ((x145[x143_2] F./ x146) F.* (x133[x143_0][x143_1][x143_2] F.+ (F.neg x136[x143_0][x143_1]))))))))
let x147 = (imap3 4 16 16 (\x148_0 x148_1 x148_2 -> (x142[x148_0][x148_1][x148_2] F./ fromi64 2)))
let x149 = (imap3 4 16 4 (\x150_0 x150_1 x150_2 -> (isum1 16 (\x151_0 -> (x128[x150_0][x151_0][x150_1] F.* x121[x150_0][x151_0][x150_2])))))
let x152 = (imap3 4 16 4 (\x153_0 x153_1 x153_2 -> (isum1 16 (\x154_0 -> (x21[x153_0][x154_0][x153_2] F.* x147[x153_0][x154_0][x153_1])))))
let x155 = (imap3 4 16 4 (\x156_0 x156_1 x156_2 -> (isum1 16 (\x157_0 -> (x147[x156_0][x156_1][x157_0] F.* x23[x156_0][x157_0][x156_2])))))
let x158 = (imap2 16 16 (\x159_0 x159_1 -> x149[(x159_1 / 4)][x159_0][(x159_1 % 4)]))
let x160 = (imap2 16 16 (\x161_0 x161_1 -> x152[(x161_1 / 4)][x161_0][(x161_1 % 4)]))
let x162 = (imap2 16 16 (\x163_0 x163_1 -> x155[(x163_1 / 4)][x163_0][(x163_1 % 4)]))
let x164 = (imap2 16 16 (\x165_0 x165_1 -> (((isum1 16 (\x166_0 -> (wval[x166_0][x165_1] F.* x158[x165_0][x166_0]))) F.+ (isum1 16 (\x167_0 -> (wkey[x167_0][x165_1] F.* x160[x165_0][x167_0])))) F.+ (isum1 16 (\x168_0 -> (wqry[x168_0][x165_1] F.* x162[x165_0][x168_0]))))))
let x169 = (imap2 16 16 (\x170_0 x170_1 -> (x2[x170_0][x170_1] F.* x2[x170_0][x170_1])))
let x171 = (imap1 16 (\x172_0 -> ((isum1 16 (\x173_0 -> x169[x172_0][x173_0])) F./ fromi64 16)))
let x174 = (imap1 16 (\x175_0 -> (F.sqrt (x171[x175_0] F.+ (one F./ fromi64 100000)))))
let x176 = (imap1 16 (\x177_0 -> (F.neg ((one F./ x174[x177_0]) F.* ((isum1 16 (\x178_0 -> (x2[x177_0][x178_0] F.* x164[x177_0][x178_0]))) F.* (one F./ x174[x177_0]))))))
let x179 = (imap1 16 (\x180_0 -> (x176[x180_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x171[x180_0] F.+ (one F./ fromi64 100000))))))))
let x181 = (imap2 16 16 (\x182_0 x182_1 -> (x179[x182_0] F./ fromi64 16)))
let x183 = (imap2 16 16 (\x184_0 x184_1 -> (x116[x184_0][x184_1] F.+ (((x164[x184_0][x184_1] F.* (one F./ x174[x184_0])) F.+ (x2[x184_0][x184_1] F.* x181[x184_0][x184_1])) F.+ (x181[x184_0][x184_1] F.* x2[x184_0][x184_1])))))
let x185 = (imap2 16 16 (\x186_0 x186_1 -> (x0[x186_0][x186_1] F.* x0[x186_0][x186_1])))
let x187 = (imap1 16 (\x188_0 -> ((isum1 16 (\x189_0 -> x185[x188_0][x189_0])) F./ fromi64 16)))
let x190 = (imap1 16 (\x191_0 -> (F.sqrt (x187[x191_0] F.+ (one F./ fromi64 100000)))))
let x192 = (imap1 16 (\x193_0 -> (F.neg ((one F./ x190[x193_0]) F.* ((isum1 16 (\x194_0 -> (x0[x193_0][x194_0] F.* x183[x193_0][x194_0]))) F.* (one F./ x190[x193_0]))))))
let x195 = (imap1 16 (\x196_0 -> (x192[x196_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x187[x196_0] F.+ (one F./ fromi64 100000))))))))
let x197 = (imap2 16 16 (\x198_0 x198_1 -> (x195[x198_0] F./ fromi64 16)))
let x199 = (imap2 16 16 (\x200_0 x200_1 -> (((x183[x200_0][x200_1] F.* (one F./ x190[x200_0])) F.+ (x0[x200_0][x200_1] F.* x197[x200_0][x200_1])) F.+ (x197[x200_0][x200_1] F.* x0[x200_0][x200_1]))))

let dmask = (imap2 16 16 (\x202_0 x202_1 -> (isum1 4 (\x201_0 -> x142[x201_0][x202_0][x202_1]))))
let dwpe = (imap2 16 16 (\x203_0 x203_1 -> x199[x203_0][x203_1]))
let dwqry = (imap2 16 16 (\x204_0 x204_1 -> (isum1 16 (\x205_0 -> (x162[x205_0][x204_0] F.* x7[x205_0][x204_1])))))
let dwkey = (imap2 16 16 (\x206_0 x206_1 -> (isum1 16 (\x207_0 -> (x160[x207_0][x206_0] F.* x7[x207_0][x206_1])))))
let dwval = (imap2 16 16 (\x208_0 x208_1 -> (isum1 16 (\x209_0 -> (x158[x209_0][x208_0] F.* x7[x209_0][x208_1])))))
let dwout = (imap2 16 16 (\x210_0 x210_1 -> (isum1 16 (\x211_0 -> (x116[x211_0][x210_0] F.* x40[x211_0][x210_1])))))
let dwup = (imap2 64 16 (\x212_0 x212_1 -> (isum1 16 (\x213_0 -> (x97[x213_0][x212_0] F.* x47[x213_0][x212_1])))))
let dwdown = (imap2 16 64 (\x214_0 x214_1 -> (isum1 16 (\x215_0 -> (x91[x215_0][x214_0] F.* x55[x215_0][x214_1])))))
let dwvoc = (imap2 27 16 (\x216_0 x216_1 -> (isum1 16 (\x217_0 -> (x83[x217_0][x216_0] F.* x60[x217_0][x216_1])))))
let dwseq = (imap2 16 16 (\x218_0 x218_1 -> x199[x218_0][x218_1]))
let dtarget = (imap2 16 27 (\x219_0 x219_1 -> (F.neg (x67[x219_0][x219_1] F.* x65[x219_0]))))


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