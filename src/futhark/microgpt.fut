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

  -- def softmax (n : i64) (logits: [n]real) : [n]real =

    -- let max_val = F.maximum  --reduce F.max F.lowest logits
    -- let exps = map (\v -> F.exp (v F.neg max_val)) logits
    -- let total = reduce (+) 0f32 exps
    -- in map (/ total) exps


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
in (let x31 = (imap2 16 16 (\x35_0 x35_1 -> (softmax x30[x35_0][x35_1])))
in (isum1 16 (\x36_0 -> (x31[x28_1][x36_0] F.* x25[x28_0][x36_0][x28_2]))))))))
let x37 = (imap2 16 16 (\x38_0 x38_1 -> x27[(x38_1 / 4)][x38_0][(x38_1 % 4)]))
let x39 = (imap2 16 16 (\x40_0 x40_1 -> (isum1 16 (\x41_0 -> (wout[x40_1][x41_0] F.* x37[x40_0][x41_0])))))
let x42 = (imap2 16 16 (\x43_0 x43_1 -> (x39[x43_0][x43_1] F.+ x2[x43_0][x43_1])))
let x44 = (imap2 16 16 (\x45_0 x45_1 -> (let x46 = ((isum1 16 (\x48_0 -> (x42[x45_0][x48_0] F.* x42[x45_0][x48_0]))) F./ fromi64 16)
in (let x47 = (F.sqrt (x46 F.+ (one F./ fromi64 100000)))
in (x42[x45_0][x45_1] F.* (one F./ x47))))))
let x49 = (imap2 16 64 (\x50_0 x50_1 -> (isum1 16 (\x51_0 -> (wup[x50_1][x51_0] F.* x44[x50_0][x51_0])))))
let x52 = (imap2 16 64 (\x53_0 x53_1 -> F.max x49[x53_0][x53_1] zero))
let x54 = (imap2 16 16 (\x55_0 x55_1 -> (isum1 64 (\x56_0 -> (wdown[x55_1][x56_0] F.* x52[x55_0][x56_0])))))
let x57 = (imap2 16 16 (\x58_0 x58_1 -> (x54[x58_0][x58_1] F.+ x42[x58_0][x58_1])))
let x59 = (imap2 16 27 (\x60_0 x60_1 -> (isum1 16 (\x61_0 -> (wvoc[x60_1][x61_0] F.* x57[x60_0][x61_0])))))
let x62 = (imap1 16 (\x63_0 -> (one F./ fromi64 16)))
let x64 = (imap2 16 27 (\x65_0 x65_1 -> (F.log (softmax x59[x65_0][x65_1]))))
let x66 = (imap2 16 27 (\x67_0 x67_1 -> ((F.neg x62[x67_0]) F.* target[x67_0][x67_1])))
let x68 = (imap1 16 (\x69_0 -> (isum1 27 (\x70_0 -> ((x66[x69_0][x70_0] F.* (one F./ (softmax x59[x69_0][x70_0]))) F.* (softmax x59[x69_0][x70_0]))))))
let x71 = (imap2 16 27 (\x72_0 x72_1 -> ((softmax x59[x72_0][x72_1]) F.* ((x66[x72_0][x72_1] F.* (one F./ (softmax x59[x72_0][x72_1]))) F.+ (F.neg x68[x72_0])))))
let x73 = (imap2 16 16 (\x74_0 x74_1 -> (isum1 27 (\x75_0 -> (wvoc[x75_0][x74_1] F.* x71[x74_0][x75_0])))))
let x76 = (imap2 16 64 (\x77_0 x77_1 -> (isum1 16 (\x78_0 -> (wdown[x78_0][x77_1] F.* x73[x77_0][x78_0])))))
let x79 = (imap2 16 64 (\x80_0 x80_1 -> ((indicatorp x49[x80_0][x80_1]) F.* x76[x80_0][x80_1])))
let x81 = (imap2 16 16 (\x82_0 x82_1 -> (isum1 64 (\x83_0 -> (wup[x83_0][x82_1] F.* x79[x82_0][x83_0])))))
let x84 = (imap2 16 16 (\x85_0 x85_1 -> (x42[x85_0][x85_1] F.* x42[x85_0][x85_1])))
let x86 = (imap1 16 (\x87_0 -> ((isum1 16 (\x88_0 -> x84[x87_0][x88_0])) F./ fromi64 16)))
let x89 = (imap1 16 (\x90_0 -> (F.sqrt (x86[x90_0] F.+ (one F./ fromi64 100000)))))
let x91 = (imap1 16 (\x92_0 -> (F.neg ((one F./ x89[x92_0]) F.* ((isum1 16 (\x93_0 -> (x42[x92_0][x93_0] F.* x81[x92_0][x93_0]))) F.* (one F./ x89[x92_0]))))))
let x94 = (imap1 16 (\x95_0 -> (x91[x95_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x86[x95_0] F.+ (one F./ fromi64 100000))))))))
let x96 = (imap2 16 16 (\x97_0 x97_1 -> (x94[x97_0] F./ fromi64 16)))
let x98 = (imap2 16 16 (\x99_0 x99_1 -> (x73[x99_0][x99_1] F.+ (((x81[x99_0][x99_1] F.* (one F./ x89[x99_0])) F.+ (x42[x99_0][x99_1] F.* x96[x99_0][x99_1])) F.+ (x96[x99_0][x99_1] F.* x42[x99_0][x99_1])))))
let x100 = (imap2 16 16 (\x101_0 x101_1 -> (isum1 16 (\x102_0 -> (wout[x102_0][x101_1] F.* x98[x101_0][x102_0])))))
let x103 = (imap3 4 16 4 (\x104_0 x104_1 x104_2 -> x100[x104_1][((x104_0 * 4) + x104_2)]))
let x105 = (imap3 4 16 16 (\x106_0 x106_1 x106_2 -> (isum1 4 (\x107_0 -> (x21[x106_0][x106_1][x107_0] F.* x23[x106_0][x106_2][x107_0])))))
let x108 = (imap3 4 16 16 (\x109_0 x109_1 x109_2 -> ((x105[x109_0][x109_1][x109_2] F./ fromi64 2) F.+ mask[x109_1][x109_2])))
let x110 = (imap3 4 16 16 (\x111_0 x111_1 x111_2 -> (softmax x108[x111_0][x111_1][x111_2])))
let x112 = (imap3 4 16 16 (\x113_0 x113_1 x113_2 -> (isum1 4 (\x114_0 -> (x103[x113_0][x113_1][x114_0] F.* x25[x113_0][x113_2][x114_0])))))
let x115 = (imap2 4 16 (\x116_0 x116_1 -> (isum1 16 (\x117_0 -> (x112[x116_0][x116_1][x117_0] F.* (softmax x108[x116_0][x116_1][x117_0]))))))
let x118 = (imap3 4 16 16 (\x119_0 x119_1 x119_2 -> ((softmax x108[x119_0][x119_1][x119_2]) F.* (x112[x119_0][x119_1][x119_2] F.+ (F.neg x115[x119_0][x119_1])))))
let x120 = (imap3 4 16 16 (\x121_0 x121_1 x121_2 -> (x118[x121_0][x121_1][x121_2] F./ fromi64 2)))
let x122 = (imap3 4 16 4 (\x123_0 x123_1 x123_2 -> (isum1 16 (\x124_0 -> (x110[x123_0][x124_0][x123_1] F.* x103[x123_0][x124_0][x123_2])))))
let x125 = (imap3 4 16 4 (\x126_0 x126_1 x126_2 -> (isum1 16 (\x127_0 -> (x21[x126_0][x127_0][x126_2] F.* x120[x126_0][x127_0][x126_1])))))
let x128 = (imap3 4 16 4 (\x129_0 x129_1 x129_2 -> (isum1 16 (\x130_0 -> (x120[x129_0][x129_1][x130_0] F.* x23[x129_0][x130_0][x129_2])))))
let x131 = (imap2 16 16 (\x132_0 x132_1 -> x122[(x132_1 / 4)][x132_0][(x132_1 % 4)]))
let x133 = (imap2 16 16 (\x134_0 x134_1 -> x125[(x134_1 / 4)][x134_0][(x134_1 % 4)]))
let x135 = (imap2 16 16 (\x136_0 x136_1 -> x128[(x136_1 / 4)][x136_0][(x136_1 % 4)]))
let x137 = (imap2 16 16 (\x138_0 x138_1 -> (((isum1 16 (\x139_0 -> (wval[x139_0][x138_1] F.* x131[x138_0][x139_0]))) F.+ (isum1 16 (\x140_0 -> (wkey[x140_0][x138_1] F.* x133[x138_0][x140_0])))) F.+ (isum1 16 (\x141_0 -> (wqry[x141_0][x138_1] F.* x135[x138_0][x141_0]))))))
let x142 = (imap2 16 16 (\x143_0 x143_1 -> (x2[x143_0][x143_1] F.* x2[x143_0][x143_1])))
let x144 = (imap1 16 (\x145_0 -> ((isum1 16 (\x146_0 -> x142[x145_0][x146_0])) F./ fromi64 16)))
let x147 = (imap1 16 (\x148_0 -> (F.sqrt (x144[x148_0] F.+ (one F./ fromi64 100000)))))
let x149 = (imap1 16 (\x150_0 -> (F.neg ((one F./ x147[x150_0]) F.* ((isum1 16 (\x151_0 -> (x2[x150_0][x151_0] F.* x137[x150_0][x151_0]))) F.* (one F./ x147[x150_0]))))))
let x152 = (imap1 16 (\x153_0 -> (x149[x153_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x144[x153_0] F.+ (one F./ fromi64 100000))))))))
let x154 = (imap2 16 16 (\x155_0 x155_1 -> (x152[x155_0] F./ fromi64 16)))
let x156 = (imap2 16 16 (\x157_0 x157_1 -> (x98[x157_0][x157_1] F.+ (((x137[x157_0][x157_1] F.* (one F./ x147[x157_0])) F.+ (x2[x157_0][x157_1] F.* x154[x157_0][x157_1])) F.+ (x154[x157_0][x157_1] F.* x2[x157_0][x157_1])))))
let x158 = (imap2 16 16 (\x159_0 x159_1 -> (x0[x159_0][x159_1] F.* x0[x159_0][x159_1])))
let x160 = (imap1 16 (\x161_0 -> ((isum1 16 (\x162_0 -> x158[x161_0][x162_0])) F./ fromi64 16)))
let x163 = (imap1 16 (\x164_0 -> (F.sqrt (x160[x164_0] F.+ (one F./ fromi64 100000)))))
let x165 = (imap1 16 (\x166_0 -> (F.neg ((one F./ x163[x166_0]) F.* ((isum1 16 (\x167_0 -> (x0[x166_0][x167_0] F.* x156[x166_0][x167_0]))) F.* (one F./ x163[x166_0]))))))
let x168 = (imap1 16 (\x169_0 -> (x165[x169_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x160[x169_0] F.+ (one F./ fromi64 100000))))))))
let x170 = (imap2 16 16 (\x171_0 x171_1 -> (x168[x171_0] F./ fromi64 16)))
let x172 = (imap2 16 16 (\x173_0 x173_1 -> (((x156[x173_0][x173_1] F.* (one F./ x163[x173_0])) F.+ (x0[x173_0][x173_1] F.* x170[x173_0][x173_1])) F.+ (x170[x173_0][x173_1] F.* x0[x173_0][x173_1]))))

let dmask = (imap2 16 16 (\x175_0 x175_1 -> (isum1 4 (\x174_0 -> x118[x174_0][x175_0][x175_1]))))
let dwpe = (imap2 16 16 (\x176_0 x176_1 -> x172[x176_0][x176_1]))
let dwqry = (imap2 16 16 (\x177_0 x177_1 -> (isum1 16 (\x178_0 -> (x135[x178_0][x177_0] F.* x7[x178_0][x177_1])))))
let dwkey = (imap2 16 16 (\x179_0 x179_1 -> (isum1 16 (\x180_0 -> (x133[x180_0][x179_0] F.* x7[x180_0][x179_1])))))
let dwval = (imap2 16 16 (\x181_0 x181_1 -> (isum1 16 (\x182_0 -> (x131[x182_0][x181_0] F.* x7[x182_0][x181_1])))))
let dwout = (imap2 16 16 (\x183_0 x183_1 -> (isum1 16 (\x184_0 -> (x98[x184_0][x183_0] F.* x37[x184_0][x183_1])))))
let dwup = (imap2 64 16 (\x185_0 x185_1 -> (isum1 16 (\x186_0 -> (x79[x186_0][x185_0] F.* x44[x186_0][x185_1])))))
let dwdown = (imap2 16 64 (\x187_0 x187_1 -> (isum1 16 (\x188_0 -> (x73[x188_0][x187_0] F.* x52[x188_0][x187_1])))))
let dwvoc = (imap2 27 16 (\x189_0 x189_1 -> (isum1 16 (\x190_0 -> (x71[x190_0][x189_0] F.* x57[x190_0][x189_1])))))
let dwseq = (imap2 16 16 (\x191_0 x191_1 -> x172[x191_0][x191_1]))
let dtarget = (imap2 16 27 (\x192_0 x192_1 -> (F.neg (x64[x192_0][x192_1] F.* x62[x192_0]))))

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