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
in (let x31 = (imap2 16 16 (\x35_0 x35_1 -> (let x36 = x30[x35_0][(argmax1_0 16 (\x39_0 -> x30[x35_0][x39_0]))]
in (let x37 = (imap1 16 (\x40_0 -> (F.exp (x30[x35_0][x40_0] F.+ (F.neg x36)))))
in (let x38 = (isum1 16 (\x41_0 -> x37[x41_0]))
in (x37[x35_1] F.* (one F./ x38)))))))
in (isum1 16 (\x42_0 -> (x31[x28_1][x42_0] F.* x25[x28_0][x42_0][x28_2]))))))))
let x43 = (imap2 16 16 (\x44_0 x44_1 -> x27[(x44_1 / 4)][x44_0][(x44_1 % 4)]))
let x45 = (imap2 16 16 (\x46_0 x46_1 -> (isum1 16 (\x47_0 -> (wout[x46_1][x47_0] F.* x43[x46_0][x47_0])))))
let x48 = (imap2 16 16 (\x49_0 x49_1 -> (x45[x49_0][x49_1] F.+ x2[x49_0][x49_1])))
let x50 = (imap2 16 16 (\x51_0 x51_1 -> (let x52 = ((isum1 16 (\x54_0 -> (x48[x51_0][x54_0] F.* x48[x51_0][x54_0]))) F./ fromi64 16)
in (let x53 = (F.sqrt (x52 F.+ (one F./ fromi64 100000)))
in (x48[x51_0][x51_1] F.* (one F./ x53))))))
let x55 = (imap2 16 64 (\x56_0 x56_1 -> (isum1 16 (\x57_0 -> (wup[x56_1][x57_0] F.* x50[x56_0][x57_0])))))
let x58 = (imap2 16 64 (\x59_0 x59_1 -> F.max x55[x59_0][x59_1] zero))
let x60 = (imap2 16 16 (\x61_0 x61_1 -> (isum1 64 (\x62_0 -> (wdown[x61_1][x62_0] F.* x58[x61_0][x62_0])))))
let x63 = (imap2 16 16 (\x64_0 x64_1 -> (x60[x64_0][x64_1] F.+ x48[x64_0][x64_1])))
let x65 = (imap2 16 27 (\x66_0 x66_1 -> (isum1 16 (\x67_0 -> (wvoc[x66_1][x67_0] F.* x63[x66_0][x67_0])))))
let x68 = (imap1 16 (\x69_0 -> (one F./ fromi64 16)))
let x70 = (imap2 16 27 (\x71_0 x71_1 -> (let x72 = x65[x71_0][(argmax1_0 27 (\x75_0 -> x65[x71_0][x75_0]))]
in (let x73 = (imap1 27 (\x76_0 -> (F.exp (x65[x71_0][x76_0] F.+ (F.neg x72)))))
in (let x74 = (isum1 27 (\x77_0 -> x73[x77_0]))
in (F.log (x73[x71_1] F.* (one F./ x74))))))))
let x78 = (imap2 16 27 (\x79_0 x79_1 -> ((F.neg x68[x79_0]) F.* target[x79_0][x79_1])))
let x80 = (imap1 16 (\x81_0 -> x65[x81_0][(argmax1_0 27 (\x82_0 -> x65[x81_0][x82_0]))]))
let x83 = (imap2 16 27 (\x84_0 x84_1 -> (F.exp (x65[x84_0][x84_1] F.+ (F.neg x80[x84_0])))))
let x85 = (imap1 16 (\x86_0 -> (isum1 27 (\x87_0 -> x83[x86_0][x87_0]))))
let x88 = (imap1 16 (\x89_0 -> (let x90 = (imap1 27 (\x95_0 -> (x85[x89_0] F.* x85[x89_0])))
in (F.neg (isum1 27 (\x91_0 -> (let x92 = x65[x89_0][(argmax1_0 27 (\x96_0 -> x65[x89_0][x96_0]))]
in (let x93 = (imap1 27 (\x97_0 -> (F.exp (x65[x89_0][x97_0] F.+ (F.neg x92)))))
in (let x94 = (isum1 27 (\x98_0 -> x93[x98_0]))
in (((x78[x89_0][x91_0] F.* (one F./ (x93[x91_0] F.* (one F./ x94)))) F.* x83[x89_0][x91_0]) F.* (one F./ x90[x91_0])))))))))))
let x99 = (imap2 16 27 (\x100_0 x100_1 -> (let x101 = x65[x100_0][(argmax1_0 27 (\x104_0 -> x65[x100_0][x104_0]))]
in (let x102 = (imap1 27 (\x105_0 -> (F.exp (x65[x100_0][x105_0] F.+ (F.neg x101)))))
in (let x103 = (isum1 27 (\x106_0 -> x102[x106_0]))
in (((x78[x100_0][x100_1] F.* (one F./ (x102[x100_1] F.* (one F./ x103)))) F.* (one F./ x85[x100_0])) F.+ x88[x100_0]))))))
let x107 = (imap1 16 (\x108_0 -> (F.neg (isum1 27 (\x109_0 -> ((F.exp (x65[x108_0][x109_0] F.+ (F.neg x80[x108_0]))) F.* x99[x108_0][x109_0]))))))
let x110 = (imap2 16 27 (\x111_0 x111_1 -> (((F.exp (x65[x111_0][x111_1] F.+ (F.neg x80[x111_0]))) F.* x99[x111_0][x111_1]) F.+ (if ((x111_1 == (argmax1_0 27 (\x112_0 -> x65[x111_0][x112_0])))) then x107[x111_0] else zero))))
let x113 = (imap2 16 16 (\x114_0 x114_1 -> (isum1 27 (\x115_0 -> (x110[x114_0][x115_0] F.* wvoc[x115_0][x114_1])))))
let x116 = (imap2 16 64 (\x117_0 x117_1 -> (isum1 16 (\x118_0 -> (x113[x117_0][x118_0] F.* wdown[x118_0][x117_1])))))
let x119 = (imap2 16 64 (\x120_0 x120_1 -> ((indicatorp x55[x120_0][x120_1]) F.* x116[x120_0][x120_1])))
let x121 = (imap2 16 16 (\x122_0 x122_1 -> (isum1 64 (\x123_0 -> (x119[x122_0][x123_0] F.* wup[x123_0][x122_1])))))
let x124 = (imap2 16 16 (\x125_0 x125_1 -> (x48[x125_0][x125_1] F.* x48[x125_0][x125_1])))
let x126 = (imap1 16 (\x127_0 -> ((isum1 16 (\x128_0 -> x124[x127_0][x128_0])) F./ fromi64 16)))
let x129 = (imap1 16 (\x130_0 -> (F.sqrt (x126[x130_0] F.+ (one F./ fromi64 100000)))))
let x131 = (imap1 16 (\x132_0 -> (let x133 = (imap1 16 (\x135_0 -> (x129[x132_0] F.* x129[x132_0])))
in (F.neg (isum1 16 (\x134_0 -> ((x121[x132_0][x134_0] F.* x48[x132_0][x134_0]) F.* (one F./ x133[x134_0]))))))))
let x136 = (imap1 16 (\x137_0 -> (x131[x137_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x126[x137_0] F.+ (one F./ fromi64 100000))))))))
let x138 = (imap2 16 16 (\x139_0 x139_1 -> (x136[x139_0] F./ fromi64 16)))
let x140 = (imap2 16 16 (\x141_0 x141_1 -> (x113[x141_0][x141_1] F.+ (((x121[x141_0][x141_1] F.* (one F./ x129[x141_0])) F.+ (x138[x141_0][x141_1] F.* x48[x141_0][x141_1])) F.+ (x138[x141_0][x141_1] F.* x48[x141_0][x141_1])))))
let x142 = (imap2 16 16 (\x143_0 x143_1 -> (isum1 16 (\x144_0 -> (x140[x143_0][x144_0] F.* wout[x144_0][x143_1])))))
let x145 = (imap3 4 16 4 (\x146_0 x146_1 x146_2 -> x142[x146_1][((x146_0 * 4) + x146_2)]))
let x147 = (imap3 4 16 16 (\x148_0 x148_1 x148_2 -> (isum1 4 (\x149_0 -> (x21[x148_0][x148_1][x149_0] F.* x23[x148_0][x148_2][x149_0])))))
let x150 = (imap3 4 16 16 (\x151_0 x151_1 x151_2 -> ((x147[x151_0][x151_1][x151_2] F./ fromi64 2) F.+ mask[x151_1][x151_2])))
let x152 = (imap3 4 16 16 (\x153_0 x153_1 x153_2 -> (let x154 = x150[x153_0][x153_1][(argmax1_0 16 (\x157_0 -> x150[x153_0][x153_1][x157_0]))]
in (let x155 = (imap1 16 (\x158_0 -> (F.exp (x150[x153_0][x153_1][x158_0] F.+ (F.neg x154)))))
in (let x156 = (isum1 16 (\x159_0 -> x155[x159_0]))
in (x155[x153_2] F.* (one F./ x156)))))))
let x160 = (imap3 4 16 16 (\x161_0 x161_1 x161_2 -> (isum1 4 (\x162_0 -> (x145[x161_0][x161_1][x162_0] F.* x25[x161_0][x161_2][x162_0])))))
let x163 = (imap2 4 16 (\x164_0 x164_1 -> x150[x164_0][x164_1][(argmax1_0 16 (\x165_0 -> x150[x164_0][x164_1][x165_0]))]))
let x166 = (imap3 4 16 16 (\x167_0 x167_1 x167_2 -> (F.exp (x150[x167_0][x167_1][x167_2] F.+ (F.neg x163[x167_0][x167_1])))))
let x168 = (imap2 4 16 (\x169_0 x169_1 -> (isum1 16 (\x170_0 -> x166[x169_0][x169_1][x170_0]))))
let x171 = (imap2 4 16 (\x172_0 x172_1 -> (let x173 = (imap1 16 (\x175_0 -> (x168[x172_0][x172_1] F.* x168[x172_0][x172_1])))
in (F.neg (isum1 16 (\x174_0 -> ((x160[x172_0][x172_1][x174_0] F.* x166[x172_0][x172_1][x174_0]) F.* (one F./ x173[x174_0]))))))))
let x176 = (imap3 4 16 16 (\x177_0 x177_1 x177_2 -> ((x160[x177_0][x177_1][x177_2] F.* (one F./ x168[x177_0][x177_1])) F.+ x171[x177_0][x177_1])))
let x178 = (imap2 4 16 (\x179_0 x179_1 -> (F.neg (isum1 16 (\x180_0 -> ((F.exp (x150[x179_0][x179_1][x180_0] F.+ (F.neg x163[x179_0][x179_1]))) F.* x176[x179_0][x179_1][x180_0]))))))
let x181 = (imap3 4 16 16 (\x182_0 x182_1 x182_2 -> (((F.exp (x150[x182_0][x182_1][x182_2] F.+ (F.neg x163[x182_0][x182_1]))) F.* x176[x182_0][x182_1][x182_2]) F.+ (if ((x182_2 == (argmax1_0 16 (\x183_0 -> x150[x182_0][x182_1][x183_0])))) then x178[x182_0][x182_1] else zero))))
let x184 = (imap3 4 16 16 (\x185_0 x185_1 x185_2 -> (x181[x185_0][x185_1][x185_2] F./ fromi64 2)))
let x186 = (imap3 4 16 4 (\x187_0 x187_1 x187_2 -> (isum1 16 (\x188_0 -> (x145[x187_0][x188_0][x187_2] F.* x152[x187_0][x188_0][x187_1])))))
let x189 = (imap3 4 16 4 (\x190_0 x190_1 x190_2 -> (isum1 16 (\x191_0 -> (x184[x190_0][x191_0][x190_1] F.* x21[x190_0][x191_0][x190_2])))))
let x192 = (imap3 4 16 4 (\x193_0 x193_1 x193_2 -> (isum1 16 (\x194_0 -> (x184[x193_0][x193_1][x194_0] F.* x23[x193_0][x194_0][x193_2])))))
let x195 = (imap2 16 16 (\x196_0 x196_1 -> x186[(x196_1 / 4)][x196_0][(x196_1 % 4)]))
let x197 = (imap2 16 16 (\x198_0 x198_1 -> x189[(x198_1 / 4)][x198_0][(x198_1 % 4)]))
let x199 = (imap2 16 16 (\x200_0 x200_1 -> x192[(x200_1 / 4)][x200_0][(x200_1 % 4)]))
let x201 = (imap2 16 16 (\x202_0 x202_1 -> (((isum1 16 (\x203_0 -> (x195[x202_0][x203_0] F.* wval[x203_0][x202_1]))) F.+ (isum1 16 (\x204_0 -> (x197[x202_0][x204_0] F.* wkey[x204_0][x202_1])))) F.+ (isum1 16 (\x205_0 -> (x199[x202_0][x205_0] F.* wqry[x205_0][x202_1]))))))
let x206 = (imap2 16 16 (\x207_0 x207_1 -> (x2[x207_0][x207_1] F.* x2[x207_0][x207_1])))
let x208 = (imap1 16 (\x209_0 -> ((isum1 16 (\x210_0 -> x206[x209_0][x210_0])) F./ fromi64 16)))
let x211 = (imap1 16 (\x212_0 -> (F.sqrt (x208[x212_0] F.+ (one F./ fromi64 100000)))))
let x213 = (imap1 16 (\x214_0 -> (let x215 = (imap1 16 (\x217_0 -> (x211[x214_0] F.* x211[x214_0])))
in (F.neg (isum1 16 (\x216_0 -> ((x201[x214_0][x216_0] F.* x2[x214_0][x216_0]) F.* (one F./ x215[x216_0]))))))))
let x218 = (imap1 16 (\x219_0 -> (x213[x219_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x208[x219_0] F.+ (one F./ fromi64 100000))))))))
let x220 = (imap2 16 16 (\x221_0 x221_1 -> (x218[x221_0] F./ fromi64 16)))
let x222 = (imap2 16 16 (\x223_0 x223_1 -> (x140[x223_0][x223_1] F.+ (((x201[x223_0][x223_1] F.* (one F./ x211[x223_0])) F.+ (x220[x223_0][x223_1] F.* x2[x223_0][x223_1])) F.+ (x220[x223_0][x223_1] F.* x2[x223_0][x223_1])))))
let x224 = (imap2 16 16 (\x225_0 x225_1 -> (x0[x225_0][x225_1] F.* x0[x225_0][x225_1])))
let x226 = (imap1 16 (\x227_0 -> ((isum1 16 (\x228_0 -> x224[x227_0][x228_0])) F./ fromi64 16)))
let x229 = (imap1 16 (\x230_0 -> (F.sqrt (x226[x230_0] F.+ (one F./ fromi64 100000)))))
let x231 = (imap1 16 (\x232_0 -> (let x233 = (imap1 16 (\x235_0 -> (x229[x232_0] F.* x229[x232_0])))
in (F.neg (isum1 16 (\x234_0 -> ((x222[x232_0][x234_0] F.* x0[x232_0][x234_0]) F.* (one F./ x233[x234_0]))))))))
let x236 = (imap1 16 (\x237_0 -> (x231[x237_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x226[x237_0] F.+ (one F./ fromi64 100000))))))))
let x238 = (imap2 16 16 (\x239_0 x239_1 -> (x236[x239_0] F./ fromi64 16)))
let x240 = (imap2 16 16 (\x241_0 x241_1 -> (((x222[x241_0][x241_1] F.* (one F./ x229[x241_0])) F.+ (x238[x241_0][x241_1] F.* x0[x241_0][x241_1])) F.+ (x238[x241_0][x241_1] F.* x0[x241_0][x241_1]))))

-- let dmask = (imap2 16 16 (\x243_0 x243_1 -> (isum1 4 (\x242_0 -> x181[x242_0][x243_0][x243_1]))))
let dwpe = (imap2 16 16 (\x244_0 x244_1 -> x240[x244_0][x244_1]))
let dwqry = (imap2 16 16 (\x245_0 x245_1 -> (isum1 16 (\x246_0 -> (x199[x246_0][x245_0] F.* x7[x246_0][x245_1])))))
let dwkey = (imap2 16 16 (\x247_0 x247_1 -> (isum1 16 (\x248_0 -> (x197[x248_0][x247_0] F.* x7[x248_0][x247_1])))))
let dwval = (imap2 16 16 (\x249_0 x249_1 -> (isum1 16 (\x250_0 -> (x195[x250_0][x249_0] F.* x7[x250_0][x249_1])))))
let dwout = (imap2 16 16 (\x251_0 x251_1 -> (isum1 16 (\x252_0 -> (x140[x252_0][x251_0] F.* x43[x252_0][x251_1])))))
let dwup = (imap2 64 16 (\x253_0 x253_1 -> (isum1 16 (\x254_0 -> (x119[x254_0][x253_0] F.* x50[x254_0][x253_1])))))
let dwdown = (imap2 16 64 (\x255_0 x255_1 -> (isum1 16 (\x256_0 -> (x113[x256_0][x255_0] F.* x58[x256_0][x255_1])))))
let dwvoc = (imap2 27 16 (\x257_0 x257_1 -> (isum1 16 (\x258_0 -> (x110[x258_0][x257_0] F.* x63[x258_0][x257_1])))))
let dwseq = (imap2 16 16 (\x259_0 x259_1 -> x240[x259_0][x259_1]))
-- let dtarget = (imap2 16 27 (\x260_0 x260_1 -> ((F.neg x68[x260_0]) F.* x70[x260_0][x260_1])))

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