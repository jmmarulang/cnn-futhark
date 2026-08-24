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

  def imaximum1  (m : i64) (f : i64 -> real) : (real, i64) =
    reduce
      (\(vx, ix) (vy , iy) -> if vx F.> vy then (vx, ix) else (vy, iy))
      (F.lowest, m) (zip (imap1 m f) (iota m) )

  def argmax1 : (m: i64) -> (i64 -> real) -> i64 =
    \m f -> let (_ , i) = imaximum1 m f in i

  def argmax1_0 = argmax1

  -- def imaximum1 : (m: i64) -> (i64 -> real) -> real =
  --   \m f -> F.maximum (imap1 m f)

  -- def imaximum2 : (m: i64)
  -- -> (n: i64)
  -- -> (i64 -> i64 -> real) -> real =
  --   \m n f -> F.maximum (imap1 m (\i -> imaximum1 n (f i)))

  -- def imaximum3 : (m: i64)
  -- -> (n: i64)
  -- -> (k: i64)
  -- -> (i64 -> i64 -> i64 -> real) -> real =
  --   \n m k f -> F.maximum (imap1 n (\i -> imaximum2 m k (f i)))

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
in (let x44 = (imap2 16 16 (\x48_0 x48_1 -> (let x49 = x43[x48_0][(argmax1_0 16 (\x52_0 -> x43[x48_0][x52_0]))]
in (let x50 = (imap1 16 (\x53_0 -> (F.exp (x43[x48_0][x53_0] F.+ (F.neg x49)))))
in (let x51 = (isum1 16 (\x54_0 -> x50[x54_0]))
in (x50[x48_1] F.* (one F./ x51)))))))
in (isum1 16 (\x55_0 -> (x44[x41_1][x55_0] F.* x8[x41_0][x55_0][x41_2]))))))))
in (let x10 = (imap2 16 16 (\x56_0 x56_1 -> x9[(x56_1 / 4)][x56_0][(x56_1 % 4)]))
in (let x11 = (imap2 16 16 (\x57_0 x57_1 -> (isum1 16 (\x58_0 -> (wout[x57_1][x58_0] F.* x10[x57_0][x58_0])))))
in (let x12 = (imap2 16 16 (\x59_0 x59_1 -> (x11[x59_0][x59_1] F.+ x1[x59_0][x59_1])))
in (let x13 = (imap2 16 16 (\x60_0 x60_1 -> (let x61 = (imap1 16 (\x64_0 -> (x12[x60_0][x64_0] F.* x12[x60_0][x64_0])))
in (let x62 = ((isum1 16 (\x65_0 -> x61[x65_0])) F./ fromi64 16)
in (let x63 = (F.sqrt (x62 F.+ (one F./ fromi64 100000)))
in (x12[x60_0][x60_1] F.* (one F./ x63)))))))
in (let x14 = (imap2 16 64 (\x66_0 x66_1 -> (isum1 16 (\x67_0 -> (wup[x66_1][x67_0] F.* x13[x66_0][x67_0])))))
in (let x15 = (imap2 16 64 (\x68_0 x68_1 -> F.max x14[x68_0][x68_1] zero))
in (let x16 = (imap2 16 16 (\x69_0 x69_1 -> (isum1 64 (\x70_0 -> (wdown[x69_1][x70_0] F.* x15[x69_0][x70_0])))))
in (let x17 = (imap2 16 16 (\x71_0 x71_1 -> (x16[x71_0][x71_1] F.+ x12[x71_0][x71_1])))
in (imap2 16 27 (\x18_0 x18_1 -> (isum1 16 (\x72_0 -> (wvoc[x18_1][x72_0] F.* x17[x18_0][x72_0])))))))))))))))))))))))


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
let x2 = (imap2 16 16 (\x3_0 x3_1 -> ((one F./ (F.sqrt (((isum1 16 (\x4_0 -> (x0[x3_0][x4_0] F.* x0[x3_0][x4_0]))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))) F.* x0[x3_0][x3_1])))
let x5 = (imap2 16 16 (\x6_0 x6_1 -> ((one F./ (F.sqrt (((isum1 16 (\x7_0 -> (x2[x6_0][x7_0] F.* x2[x6_0][x7_0]))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))) F.* x2[x6_0][x6_1])))
let x8 = (imap2 16 16 (\x9_0 x9_1 -> (isum1 16 (\x10_0 -> (wqry[x9_1][x10_0] F.* x5[x9_0][x10_0])))))
let x11 = (imap2 16 16 (\x12_0 x12_1 -> (isum1 16 (\x13_0 -> (wkey[x12_1][x13_0] F.* x5[x12_0][x13_0])))))
let x14 = (imap2 16 16 (\x15_0 x15_1 -> (isum1 16 (\x16_0 -> (wval[x15_1][x16_0] F.* x5[x15_0][x16_0])))))
let x17 = (imap3 4 16 4 (\x18_0 x18_1 x18_2 -> x8[x18_1][((x18_0 * 4) + x18_2)]))
let x19 = (imap3 4 16 4 (\x20_0 x20_1 x20_2 -> x11[x20_1][((x20_0 * 4) + x20_2)]))
let x21 = (imap3 4 16 4 (\x22_0 x22_1 x22_2 -> x14[x22_1][((x22_0 * 4) + x22_2)]))
let x23 = (imap3 4 16 4 (\x24_0 x24_1 x24_2 -> (let x25 = (imap2 16 16 (\x26_0 x26_1 -> (((isum1 4 (\x27_0 -> (x17[x24_0][x26_0][x27_0] F.* x19[x24_0][x26_1][x27_0]))) F./ fromi64 2) F.+ mask[x26_0][x26_1])))
in (let x28 = (imap1 16 (\x29_0 -> (F.exp (x25[x24_1][x29_0] F.+ (F.neg x25[x24_1][(argmax1_0 16 (\x30_0 -> x25[x24_1][x30_0]))])))))
in (isum1 16 (\x31_0 -> (((one F./ (isum1 16 (\x32_0 -> x28[x32_0]))) F.* x28[x31_0]) F.* x21[x24_0][x31_0][x24_2])))))))
let x33 = (imap2 16 16 (\x34_0 x34_1 -> x23[(x34_1 / 4)][x34_0][(x34_1 % 4)]))
let x35 = (imap2 16 16 (\x36_0 x36_1 -> (isum1 16 (\x37_0 -> (wout[x36_1][x37_0] F.* x33[x36_0][x37_0])))))
let x38 = (imap2 16 16 (\x39_0 x39_1 -> (x35[x39_0][x39_1] F.+ x2[x39_0][x39_1])))
let x40 = (imap2 16 16 (\x41_0 x41_1 -> ((one F./ (F.sqrt (((isum1 16 (\x42_0 -> (x38[x41_0][x42_0] F.* x38[x41_0][x42_0]))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))) F.* x38[x41_0][x41_1])))
let x43 = (imap2 16 64 (\x44_0 x44_1 -> (isum1 16 (\x45_0 -> (wup[x44_1][x45_0] F.* x40[x44_0][x45_0])))))
let x46 = (imap2 16 64 (\x47_0 x47_1 -> F.max x43[x47_0][x47_1] zero))
let x48 = (imap2 16 16 (\x49_0 x49_1 -> (isum1 64 (\x50_0 -> (wdown[x49_1][x50_0] F.* x46[x49_0][x50_0])))))
let x51 = (imap2 16 16 (\x52_0 x52_1 -> (x48[x52_0][x52_1] F.+ x38[x52_0][x52_1])))
let x53 = (imap2 16 27 (\x54_0 x54_1 -> (isum1 16 (\x55_0 -> (wvoc[x54_1][x55_0] F.* x51[x54_0][x55_0])))))
let x56 = (imap1 16 (\x57_0 -> (one F./ fromi64 16)))
let x58 = (imap2 16 27 (\x59_0 x59_1 -> (let x60 = (imap1 27 (\x61_0 -> (F.exp (x53[x59_0][x61_0] F.+ (F.neg x53[x59_0][(argmax1_0 27 (\x62_0 -> x53[x59_0][x62_0]))])))))
in (F.log ((one F./ (isum1 27 (\x63_0 -> x60[x63_0]))) F.* x60[x59_1])))))
let x64 = (imap2 16 27 (\x65_0 x65_1 -> ((F.neg x56[x65_0]) F.* target[x65_0][x65_1])))
let x66 = (imap1 16 (\x67_0 -> x53[x67_0][(argmax1_0 27 (\x68_0 -> x53[x67_0][x68_0]))]))
let x69 = (imap2 16 27 (\x70_0 x70_1 -> (x53[x70_0][x70_1] F.+ (F.neg x66[x70_0]))))
let x71 = (imap2 16 27 (\x72_0 x72_1 -> (F.exp x69[x72_0][x72_1])))
let x73 = (imap1 16 (\x74_0 -> (isum1 27 (\x75_0 -> x71[x74_0][x75_0]))))
let x76 = (imap1 16 (\x77_0 -> (F.neg ((one F./ x73[x77_0]) F.* ((one F./ x73[x77_0]) F.* (isum1 27 (\x78_0 -> (let x79 = (imap1 27 (\x81_0 -> (F.exp (x53[x77_0][x81_0] F.+ (F.neg x53[x77_0][(argmax1_0 27 (\x82_0 -> x53[x77_0][x82_0]))])))))
in ((one F./ ((one F./ (isum1 27 (\x80_0 -> x79[x80_0]))) F.* x79[x78_0])) F.* (x71[x77_0][x78_0] F.* x64[x77_0][x78_0]))))))))))
let x83 = (imap2 16 27 (\x84_0 x84_1 -> (let x85 = (imap1 27 (\x87_0 -> (F.exp (x53[x84_0][x87_0] F.+ (F.neg x53[x84_0][(argmax1_0 27 (\x88_0 -> x53[x84_0][x88_0]))])))))
in (((one F./ x73[x84_0]) F.* ((one F./ ((one F./ (isum1 27 (\x86_0 -> x85[x86_0]))) F.* x85[x84_1])) F.* x64[x84_0][x84_1])) F.+ x76[x84_0]))))
let x89 = (imap2 16 27 (\x90_0 x90_1 -> ((F.exp x69[x90_0][x90_1]) F.* x83[x90_0][x90_1])))
let x91 = (imap1 16 (\x92_0 -> (F.neg (isum1 27 (\x93_0 -> x89[x92_0][x93_0])))))
let x94 = (imap2 16 27 (\x95_0 x95_1 -> (if ((x95_1 == (argmax1_0 27 (\x96_0 -> x53[x95_0][x96_0])))) then x91[x95_0] else zero)))
let x97 = (imap2 16 27 (\x98_0 x98_1 -> (x89[x98_0][x98_1] F.+ x94[x98_0][x98_1])))
let x99 = (imap2 16 16 (\x100_0 x100_1 -> (isum1 27 (\x101_0 -> (wvoc[x101_0][x100_1] F.* x97[x100_0][x101_0])))))
let x102 = (imap2 16 64 (\x103_0 x103_1 -> (isum1 16 (\x104_0 -> (wdown[x104_0][x103_1] F.* x99[x103_0][x104_0])))))
let x105 = (imap2 16 64 (\x106_0 x106_1 -> ((indicatorp x43[x106_0][x106_1]) F.* x102[x106_0][x106_1])))
let x107 = (imap2 16 16 (\x108_0 x108_1 -> (isum1 64 (\x109_0 -> (wup[x109_0][x108_1] F.* x105[x108_0][x109_0])))))
let x110 = (imap2 16 16 (\x111_0 x111_1 -> (x38[x111_0][x111_1] F.* x38[x111_0][x111_1])))
let x112 = (imap1 16 (\x113_0 -> ((isum1 16 (\x114_0 -> x110[x113_0][x114_0])) F./ fromi64 16)))
let x115 = (imap1 16 (\x116_0 -> (F.sqrt (x112[x116_0] F.+ (one F./ fromi64 100000)))))
let x117 = (imap1 16 (\x118_0 -> (F.neg ((one F./ x115[x118_0]) F.* ((one F./ x115[x118_0]) F.* (isum1 16 (\x119_0 -> (x38[x118_0][x119_0] F.* x107[x118_0][x119_0]))))))))
let x120 = (imap1 16 (\x121_0 -> ((one F./ ((one F.+ one) F.* (F.sqrt (x112[x121_0] F.+ (one F./ fromi64 100000))))) F.* x117[x121_0])))
let x122 = (imap2 16 16 (\x123_0 x123_1 -> (x120[x123_0] F./ fromi64 16)))
let x124 = (imap2 16 16 (\x125_0 x125_1 -> (x99[x125_0][x125_1] F.+ ((((one F./ x115[x125_0]) F.* x107[x125_0][x125_1]) F.+ (x38[x125_0][x125_1] F.* x122[x125_0][x125_1])) F.+ (x122[x125_0][x125_1] F.* x38[x125_0][x125_1])))))
let x126 = (imap2 16 16 (\x127_0 x127_1 -> (isum1 16 (\x128_0 -> (wout[x128_0][x127_1] F.* x124[x127_0][x128_0])))))
let x129 = (imap3 4 16 4 (\x130_0 x130_1 x130_2 -> x126[x130_1][((x130_0 * 4) + x130_2)]))
let x131 = (imap3 4 16 16 (\x132_0 x132_1 x132_2 -> (isum1 4 (\x133_0 -> (x17[x132_0][x132_1][x133_0] F.* x19[x132_0][x132_2][x133_0])))))
let x134 = (imap3 4 16 16 (\x135_0 x135_1 x135_2 -> ((x131[x135_0][x135_1][x135_2] F./ fromi64 2) F.+ mask[x135_1][x135_2])))
let x136 = (imap3 4 16 16 (\x137_0 x137_1 x137_2 -> (let x138 = (imap1 16 (\x139_0 -> (F.exp (x134[x137_0][x137_1][x139_0] F.+ (F.neg x134[x137_0][x137_1][(argmax1_0 16 (\x140_0 -> x134[x137_0][x137_1][x140_0]))])))))
in ((one F./ (isum1 16 (\x141_0 -> x138[x141_0]))) F.* x138[x137_2]))))
let x142 = (imap3 4 16 16 (\x143_0 x143_1 x143_2 -> (isum1 4 (\x144_0 -> (x129[x143_0][x143_1][x144_0] F.* x21[x143_0][x143_2][x144_0])))))
let x145 = (imap2 4 16 (\x146_0 x146_1 -> x134[x146_0][x146_1][(argmax1_0 16 (\x147_0 -> x134[x146_0][x146_1][x147_0]))]))
let x148 = (imap3 4 16 16 (\x149_0 x149_1 x149_2 -> (x134[x149_0][x149_1][x149_2] F.+ (F.neg x145[x149_0][x149_1]))))
let x150 = (imap3 4 16 16 (\x151_0 x151_1 x151_2 -> (F.exp x148[x151_0][x151_1][x151_2])))
let x152 = (imap2 4 16 (\x153_0 x153_1 -> (isum1 16 (\x154_0 -> x150[x153_0][x153_1][x154_0]))))
let x155 = (imap2 4 16 (\x156_0 x156_1 -> (F.neg ((one F./ x152[x156_0][x156_1]) F.* ((one F./ x152[x156_0][x156_1]) F.* (isum1 16 (\x157_0 -> (x150[x156_0][x156_1][x157_0] F.* x142[x156_0][x156_1][x157_0]))))))))
let x158 = (imap3 4 16 16 (\x159_0 x159_1 x159_2 -> (((one F./ x152[x159_0][x159_1]) F.* x142[x159_0][x159_1][x159_2]) F.+ x155[x159_0][x159_1])))
let x160 = (imap3 4 16 16 (\x161_0 x161_1 x161_2 -> ((F.exp x148[x161_0][x161_1][x161_2]) F.* x158[x161_0][x161_1][x161_2])))
let x162 = (imap2 4 16 (\x163_0 x163_1 -> (F.neg (isum1 16 (\x164_0 -> x160[x163_0][x163_1][x164_0])))))
let x165 = (imap3 4 16 16 (\x166_0 x166_1 x166_2 -> (if ((x166_2 == (argmax1_0 16 (\x167_0 -> x134[x166_0][x166_1][x167_0])))) then x162[x166_0][x166_1] else zero)))
let x168 = (imap3 4 16 16 (\x169_0 x169_1 x169_2 -> (x160[x169_0][x169_1][x169_2] F.+ x165[x169_0][x169_1][x169_2])))
let x170 = (imap3 4 16 16 (\x171_0 x171_1 x171_2 -> (x168[x171_0][x171_1][x171_2] F./ fromi64 2)))
let x172 = (imap3 4 16 4 (\x173_0 x173_1 x173_2 -> (isum1 16 (\x174_0 -> (x136[x173_0][x174_0][x173_1] F.* x129[x173_0][x174_0][x173_2])))))
let x175 = (imap3 4 16 4 (\x176_0 x176_1 x176_2 -> (isum1 16 (\x177_0 -> (x17[x176_0][x177_0][x176_2] F.* x170[x176_0][x177_0][x176_1])))))
let x178 = (imap3 4 16 4 (\x179_0 x179_1 x179_2 -> (isum1 16 (\x180_0 -> (x170[x179_0][x179_1][x180_0] F.* x19[x179_0][x180_0][x179_2])))))
let x181 = (imap2 16 16 (\x182_0 x182_1 -> x172[(x182_1 / 4)][x182_0][(x182_1 % 4)]))
let x183 = (imap2 16 16 (\x184_0 x184_1 -> x175[(x184_1 / 4)][x184_0][(x184_1 % 4)]))
let x185 = (imap2 16 16 (\x186_0 x186_1 -> x178[(x186_1 / 4)][x186_0][(x186_1 % 4)]))
let x187 = (imap2 16 16 (\x188_0 x188_1 -> (((isum1 16 (\x189_0 -> (wval[x189_0][x188_1] F.* x181[x188_0][x189_0]))) F.+ (isum1 16 (\x190_0 -> (wkey[x190_0][x188_1] F.* x183[x188_0][x190_0])))) F.+ (isum1 16 (\x191_0 -> (wqry[x191_0][x188_1] F.* x185[x188_0][x191_0]))))))
let x192 = (imap2 16 16 (\x193_0 x193_1 -> (x2[x193_0][x193_1] F.* x2[x193_0][x193_1])))
let x194 = (imap1 16 (\x195_0 -> ((isum1 16 (\x196_0 -> x192[x195_0][x196_0])) F./ fromi64 16)))
let x197 = (imap1 16 (\x198_0 -> (F.sqrt (x194[x198_0] F.+ (one F./ fromi64 100000)))))
let x199 = (imap1 16 (\x200_0 -> (F.neg ((one F./ x197[x200_0]) F.* ((one F./ x197[x200_0]) F.* (isum1 16 (\x201_0 -> (x2[x200_0][x201_0] F.* x187[x200_0][x201_0]))))))))
let x202 = (imap1 16 (\x203_0 -> ((one F./ ((one F.+ one) F.* (F.sqrt (x194[x203_0] F.+ (one F./ fromi64 100000))))) F.* x199[x203_0])))
let x204 = (imap2 16 16 (\x205_0 x205_1 -> (x202[x205_0] F./ fromi64 16)))
let x206 = (imap2 16 16 (\x207_0 x207_1 -> (x124[x207_0][x207_1] F.+ ((((one F./ x197[x207_0]) F.* x187[x207_0][x207_1]) F.+ (x2[x207_0][x207_1] F.* x204[x207_0][x207_1])) F.+ (x204[x207_0][x207_1] F.* x2[x207_0][x207_1])))))
let x208 = (imap2 16 16 (\x209_0 x209_1 -> (x0[x209_0][x209_1] F.* x0[x209_0][x209_1])))
let x210 = (imap1 16 (\x211_0 -> ((isum1 16 (\x212_0 -> x208[x211_0][x212_0])) F./ fromi64 16)))
let x213 = (imap1 16 (\x214_0 -> (F.sqrt (x210[x214_0] F.+ (one F./ fromi64 100000)))))
let x215 = (imap1 16 (\x216_0 -> (F.neg ((one F./ x213[x216_0]) F.* ((one F./ x213[x216_0]) F.* (isum1 16 (\x217_0 -> (x0[x216_0][x217_0] F.* x206[x216_0][x217_0]))))))))
let x218 = (imap1 16 (\x219_0 -> ((one F./ ((one F.+ one) F.* (F.sqrt (x210[x219_0] F.+ (one F./ fromi64 100000))))) F.* x215[x219_0])))
let x220 = (imap2 16 16 (\x221_0 x221_1 -> (x218[x221_0] F./ fromi64 16)))
let x222 = (imap2 16 16 (\x223_0 x223_1 -> ((((one F./ x213[x223_0]) F.* x206[x223_0][x223_1]) F.+ (x0[x223_0][x223_1] F.* x220[x223_0][x223_1])) F.+ (x220[x223_0][x223_1] F.* x0[x223_0][x223_1]))))

let dmask = (imap2 16 16 (\x225_0 x225_1 -> (isum1 4 (\x224_0 -> x168[x224_0][x225_0][x225_1]))))
let dwpe = (imap2 16 16 (\x226_0 x226_1 -> x222[x226_0][x226_1]))
let dwqry = (imap2 16 16 (\x227_0 x227_1 -> (isum1 16 (\x228_0 -> (x185[x228_0][x227_0] F.* x5[x228_0][x227_1])))))
let dwkey = (imap2 16 16 (\x229_0 x229_1 -> (isum1 16 (\x230_0 -> (x183[x230_0][x229_0] F.* x5[x230_0][x229_1])))))
let dwval = (imap2 16 16 (\x231_0 x231_1 -> (isum1 16 (\x232_0 -> (x181[x232_0][x231_0] F.* x5[x232_0][x231_1])))))
let dwout = (imap2 16 16 (\x233_0 x233_1 -> (isum1 16 (\x234_0 -> (x124[x234_0][x233_0] F.* x33[x234_0][x233_1])))))
let dwup = (imap2 64 16 (\x235_0 x235_1 -> (isum1 16 (\x236_0 -> (x105[x236_0][x235_0] F.* x40[x236_0][x235_1])))))
let dwdown = (imap2 16 64 (\x237_0 x237_1 -> (isum1 16 (\x238_0 -> (x99[x238_0][x237_0] F.* x46[x238_0][x237_1])))))
let dwvoc = (imap2 27 16 (\x239_0 x239_1 -> (isum1 16 (\x240_0 -> (x97[x240_0][x239_0] F.* x51[x240_0][x239_1])))))
let dwseq = (imap2 16 16 (\x241_0 x241_1 -> x222[x241_0][x241_1]))
let dtarget = (imap2 16 27 (\x242_0 x242_1 -> (F.neg (x58[x242_0][x242_1] F.* x56[x242_0]))))

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