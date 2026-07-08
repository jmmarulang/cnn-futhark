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

(let x0 = (imap2 16 16 (\x19_0 x19_1 -> (let x20 = ((isum1 16 (\x23_0 -> ((wpe[x19_0][x23_0] F.+ wseq[x19_0][x23_0]) F.* (wpe[x19_0][x23_0] F.+ wseq[x19_0][x23_0])))) F./ fromi64 16)
in (let x21 = (F.sqrt (x20 F.+ (one F./ fromi64 100000)))
in (let x22 = (imap1 16 (\x24_0 -> ((wpe[x19_0][x24_0] F.+ wseq[x19_0][x24_0]) F.* (one F./ x21))))
in x22[x19_1])))))
in (let x1 = (imap2 16 16 (\x25_0 x25_1 -> (let x26 = ((isum1 16 (\x29_0 -> (x0[x25_0][x29_0] F.* x0[x25_0][x29_0]))) F./ fromi64 16)
in (let x27 = (F.sqrt (x26 F.+ (one F./ fromi64 100000)))
in (let x28 = (imap1 16 (\x30_0 -> (x0[x25_0][x30_0] F.* (one F./ x27))))
in x28[x25_1])))))
in (let x2 = (imap2 16 16 (\x31_0 x31_1 -> (isum1 16 (\x32_0 -> (wqry[x31_1][x32_0] F.* x1[x31_0][x32_0])))))
in (let x3 = (imap2 16 16 (\x33_0 x33_1 -> (isum1 16 (\x34_0 -> (wkey[x33_1][x34_0] F.* x1[x33_0][x34_0])))))
in (let x4 = (imap2 16 16 (\x35_0 x35_1 -> (isum1 16 (\x36_0 -> (wval[x35_1][x36_0] F.* x1[x35_0][x36_0])))))
in (let x5 = (imap3 4 16 4 (\x37_0 x37_1 x37_2 -> x2[x37_1][((x37_0 * 4) + x37_2)]))
in (let x6 = (imap3 4 16 4 (\x38_0 x38_1 x38_2 -> x3[x38_1][((x38_0 * 4) + x38_2)]))
in (let x7 = (imap3 4 16 4 (\x39_0 x39_1 x39_2 -> x4[x39_1][((x39_0 * 4) + x39_2)]))
in (let x8 = (imap3 4 16 4 (\x40_0 x40_1 x40_2 -> (let x41 = (imap2 16 16 (\x45_0 x45_1 -> (isum1 4 (\x46_0 -> (x5[x40_0][x45_0][x46_0] F.* x6[x40_0][x45_1][x46_0])))))
in (let x42 = (imap2 16 16 (\x47_0 x47_1 -> ((x41[x47_0][x47_1] F./ fromi64 2) F.+ mask[x47_0][x47_1])))
in (let x43 = (imap2 16 16 (\x48_0 x48_1 -> (let x49 = (imap1 16 (\x52_0 -> (F.exp x42[x48_0][x52_0])))
in (let x50 = (isum1 16 (\x53_0 -> x49[x53_0]))
in (let x51 = (imap1 16 (\x54_0 -> (x49[x54_0] F.* (one F./ x50))))
in x51[x48_1])))))
in (let x44 = (imap2 16 4 (\x55_0 x55_1 -> (isum1 16 (\x56_0 -> (x43[x55_0][x56_0] F.* x7[x40_0][x56_0][x55_1])))))
in x44[x40_1][x40_2]))))))
in (let x9 = (imap2 16 16 (\x57_0 x57_1 -> x8[(x57_1 / 4)][x57_0][(x57_1 % 4)]))
in (let x10 = (imap2 16 16 (\x58_0 x58_1 -> (isum1 16 (\x59_0 -> (wout[x58_1][x59_0] F.* x9[x58_0][x59_0])))))
in (let x11 = (imap2 16 16 (\x60_0 x60_1 -> (x10[x60_0][x60_1] F.+ x0[x60_0][x60_1])))
in (let x12 = (imap2 16 16 (\x61_0 x61_1 -> (let x62 = ((isum1 16 (\x65_0 -> (x11[x61_0][x65_0] F.* x11[x61_0][x65_0]))) F./ fromi64 16)
in (let x63 = (F.sqrt (x62 F.+ (one F./ fromi64 100000)))
in (let x64 = (imap1 16 (\x66_0 -> (x11[x61_0][x66_0] F.* (one F./ x63))))
in x64[x61_1])))))
in (let x13 = (imap2 16 64 (\x67_0 x67_1 -> (isum1 16 (\x68_0 -> (wup[x67_1][x68_0] F.* x12[x67_0][x68_0])))))
in (let x14 = (imap2 16 64 (\x69_0 x69_1 -> F.max x13[x69_0][x69_1] zero))
in (let x15 = (imap2 16 16 (\x70_0 x70_1 -> (isum1 64 (\x71_0 -> (wdown[x70_1][x71_0] F.* x14[x70_0][x71_0])))))
in (let x16 = (imap2 16 16 (\x72_0 x72_1 -> (x15[x72_0][x72_1] F.+ x11[x72_0][x72_1])))
in (let x17 = (imap2 16 27 (\x73_0 x73_1 -> (isum1 16 (\x74_0 -> (wvoc[x73_1][x74_0] F.* x16[x73_0][x74_0])))))
in (imap2 16 27 (\x18_0 x18_1 -> x17[x18_0][x18_1]))))))))))))))))))))

  def cal_loss : (mask: [16][16]real)
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
    -> (real, [16]real) =
    #[unsafe]
    \(mask: [16][16]real) (wpe: [16][16]real)
    (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real)
    (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real)
    (wvoc: [27][16]real) (wseq: [16][16]real) (target: [16][27]real) ->

(let x0 = (imap2 16 16 (\x20_0 x20_1 -> (let x21 = ((isum1 16 (\x24_0 -> ((wpe[x20_0][x24_0] F.+ wseq[x20_0][x24_0]) F.* (wpe[x20_0][x24_0] F.+ wseq[x20_0][x24_0])))) F./ fromi64 16)
in (let x22 = (F.sqrt (x21 F.+ (one F./ fromi64 100000)))
in (let x23 = (imap1 16 (\x25_0 -> ((wpe[x20_0][x25_0] F.+ wseq[x20_0][x25_0]) F.* (one F./ x22))))
in x23[x20_1])))))
in (let x1 = (imap2 16 16 (\x26_0 x26_1 -> (let x27 = ((isum1 16 (\x30_0 -> (x0[x26_0][x30_0] F.* x0[x26_0][x30_0]))) F./ fromi64 16)
in (let x28 = (F.sqrt (x27 F.+ (one F./ fromi64 100000)))
in (let x29 = (imap1 16 (\x31_0 -> (x0[x26_0][x31_0] F.* (one F./ x28))))
in x29[x26_1])))))
in (let x2 = (imap2 16 16 (\x32_0 x32_1 -> (isum1 16 (\x33_0 -> (wqry[x32_1][x33_0] F.* x1[x32_0][x33_0])))))
in (let x3 = (imap2 16 16 (\x34_0 x34_1 -> (isum1 16 (\x35_0 -> (wkey[x34_1][x35_0] F.* x1[x34_0][x35_0])))))
in (let x4 = (imap2 16 16 (\x36_0 x36_1 -> (isum1 16 (\x37_0 -> (wval[x36_1][x37_0] F.* x1[x36_0][x37_0])))))
in (let x5 = (imap3 4 16 4 (\x38_0 x38_1 x38_2 -> x2[x38_1][((x38_0 * 4) + x38_2)]))
in (let x6 = (imap3 4 16 4 (\x39_0 x39_1 x39_2 -> x3[x39_1][((x39_0 * 4) + x39_2)]))
in (let x7 = (imap3 4 16 4 (\x40_0 x40_1 x40_2 -> x4[x40_1][((x40_0 * 4) + x40_2)]))
in (let x8 = (imap3 4 16 4 (\x41_0 x41_1 x41_2 -> (isum1 16 (\x42_0 -> (((F.exp (((isum1 4 (\x44_0 -> (x5[x41_0][x41_1][x44_0] F.* x6[x41_0][x42_0][x44_0]))) F./ fromi64 2) F.+ mask[x41_1][x42_0])) F.* (one F./ (isum1 16 (\x43_0 -> (F.exp (((isum1 4 (\x45_0 -> (x5[x41_0][x41_1][x45_0] F.* x6[x41_0][x43_0][x45_0]))) F./ fromi64 2) F.+ mask[x41_1][x43_0])))))) F.* x7[x41_0][x42_0][x41_2])))))
in (let x9 = (imap2 16 16 (\x46_0 x46_1 -> x8[(x46_1 / 4)][x46_0][(x46_1 % 4)]))
in (let x10 = (imap2 16 16 (\x47_0 x47_1 -> (isum1 16 (\x48_0 -> (wout[x47_1][x48_0] F.* x9[x47_0][x48_0])))))
in (let x11 = (imap2 16 16 (\x49_0 x49_1 -> (x10[x49_0][x49_1] F.+ x0[x49_0][x49_1])))
in (let x12 = (imap2 16 16 (\x50_0 x50_1 -> (let x51 = ((isum1 16 (\x54_0 -> (x11[x50_0][x54_0] F.* x11[x50_0][x54_0]))) F./ fromi64 16)
in (let x52 = (F.sqrt (x51 F.+ (one F./ fromi64 100000)))
in (let x53 = (imap1 16 (\x55_0 -> (x11[x50_0][x55_0] F.* (one F./ x52))))
in x53[x50_1])))))
in (let x13 = (imap2 16 64 (\x56_0 x56_1 -> (isum1 16 (\x57_0 -> (wup[x56_1][x57_0] F.* x12[x56_0][x57_0])))))
in (let x14 = (imap2 16 64 (\x58_0 x58_1 -> F.max x13[x58_0][x58_1] zero))
in (let x15 = (imap2 16 16 (\x59_0 x59_1 -> (isum1 64 (\x60_0 -> (wdown[x59_1][x60_0] F.* x14[x59_0][x60_0])))))
in (let x16 = (imap2 16 16 (\x61_0 x61_1 -> (x15[x61_0][x61_1] F.+ x11[x61_0][x61_1])))
in (let x17 = (imap2 16 27 (\x62_0 x62_1 -> (isum1 16 (\x63_0 -> (wvoc[x62_1][x63_0] F.* x16[x62_0][x63_0])))))
in (let x18 = (imap1 16 (\x64_0 -> (F.neg (isum1 27 (\x65_0 -> ((F.log ((F.exp x17[x64_0][x65_0]) F.* (one F./ (isum1 27 (\x66_0 -> (F.exp x17[x64_0][x66_0])))))) F.* target[x64_0][x65_0]))))))
in (let x19 = ((isum1 16 (\x67_0 -> x18[x67_0])) F./ fromi64 16)
in
--x19
let loss = x19
let losses = x18
in (loss, losses)
))))))))))))))))))))



  -- is this correct? does it fill with zeroes if sl < 16?
  -- def cal_target [asl] : (target_ids : [asl]i64) -> [16][27]real =
  --   \(target_ids : [asl]i64) ->
  --   imap2 16 27 (\n m -> (if (n < asl && target_ids[n] == m) then one else zero))

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

let x0 = (imap2 16 16 (\x1_0 x1_1 -> (let x2 = ((isum1 16 (\x5_0 -> ((wpe[x1_0][x5_0] F.+ wseq[x1_0][x5_0]) F.* (wpe[x1_0][x5_0] F.+ wseq[x1_0][x5_0])))) F./ fromi64 16)
in (let x3 = (F.sqrt (x2 F.+ (one F./ fromi64 100000)))
in (let x4 = (imap1 16 (\x6_0 -> ((wpe[x1_0][x6_0] F.+ wseq[x1_0][x6_0]) F.* (one F./ x3))))
in x4[x1_1])))))
let x7 = (imap2 16 16 (\x8_0 x8_1 -> (let x9 = ((isum1 16 (\x12_0 -> (x0[x8_0][x12_0] F.* x0[x8_0][x12_0]))) F./ fromi64 16)
in (let x10 = (F.sqrt (x9 F.+ (one F./ fromi64 100000)))
in (let x11 = (imap1 16 (\x13_0 -> (x0[x8_0][x13_0] F.* (one F./ x10))))
in x11[x8_1])))))
let x14 = (imap3 4 16 4 (\x15_0 x15_1 x15_2 -> (isum1 16 (\x16_0 -> (wqry[((x15_0 * 4) + x15_2)][x16_0] F.* x7[x15_1][x16_0])))))
let x17 = (imap3 4 16 4 (\x18_0 x18_1 x18_2 -> (isum1 16 (\x19_0 -> (wkey[((x18_0 * 4) + x18_2)][x19_0] F.* x7[x18_1][x19_0])))))
let x20 = (imap3 4 16 4 (\x21_0 x21_1 x21_2 -> (isum1 16 (\x22_0 -> (wval[((x21_0 * 4) + x21_2)][x22_0] F.* x7[x21_1][x22_0])))))
let x23 = (imap2 16 16 (\x24_0 x24_1 -> (let x25 = (imap2 16 16 (\x29_0 x29_1 -> (isum1 4 (\x30_0 -> (x14[(x24_1 / 4)][x29_0][x30_0] F.* x17[(x24_1 / 4)][x29_1][x30_0])))))
in (let x26 = (imap2 16 16 (\x31_0 x31_1 -> ((x25[x31_0][x31_1] F./ fromi64 2) F.+ mask[x31_0][x31_1])))
in (let x27 = (imap2 16 16 (\x32_0 x32_1 -> (let x33 = (imap1 16 (\x36_0 -> (F.exp x26[x32_0][x36_0])))
in (let x34 = (isum1 16 (\x37_0 -> x33[x37_0]))
in (let x35 = (imap1 16 (\x38_0 -> (x33[x38_0] F.* (one F./ x34))))
in x35[x32_1])))))
in (let x28 = (imap2 16 4 (\x39_0 x39_1 -> (isum1 16 (\x40_0 -> (x27[x39_0][x40_0] F.* x20[(x24_1 / 4)][x40_0][x39_1])))))
in x28[x24_0][(x24_1 % 4)]))))))
let x41 = (imap2 16 16 (\x42_0 x42_1 -> ((isum1 16 (\x43_0 -> (wout[x42_1][x43_0] F.* x23[x42_0][x43_0]))) F.+ x0[x42_0][x42_1])))
let x44 = (imap2 16 16 (\x45_0 x45_1 -> (let x46 = ((isum1 16 (\x49_0 -> (x41[x45_0][x49_0] F.* x41[x45_0][x49_0]))) F./ fromi64 16)
in (let x47 = (F.sqrt (x46 F.+ (one F./ fromi64 100000)))
in (let x48 = (imap1 16 (\x50_0 -> (x41[x45_0][x50_0] F.* (one F./ x47))))
in x48[x45_1])))))
let x51 = (imap2 16 64 (\x52_0 x52_1 -> (isum1 16 (\x53_0 -> (wup[x52_1][x53_0] F.* x44[x52_0][x53_0])))))
let x54 = (imap2 16 64 (\x55_0 x55_1 -> F.max x51[x55_0][x55_1] zero))
let x56 = (imap2 16 16 (\x57_0 x57_1 -> ((isum1 64 (\x58_0 -> (wdown[x57_1][x58_0] F.* x54[x57_0][x58_0]))) F.+ x41[x57_0][x57_1])))
let x59 = (imap2 16 27 (\x60_0 x60_1 -> (isum1 16 (\x61_0 -> (wvoc[x60_1][x61_0] F.* x56[x60_0][x61_0])))))
let x62 = (imap1 16 (\x63_0 -> (one F./ fromi64 16)))
let x64 = (let x65 = (imap3 16 27 27 (\x71_0 x71_1 x71_2 -> (F.exp x59[x71_0][x71_2])))
in (let x66 = (imap2 16 27 (\x72_0 x72_1 -> (isum1 27 (\x73_0 -> x65[x72_0][x72_1][x73_0]))))
in (let x67 = (imap3 16 27 27 (\x74_0 x74_1 x74_2 -> (let x75 = (imap1 27 (\x78_0 -> (F.exp x59[x74_0][x78_0])))
in (let x76 = (isum1 27 (\x79_0 -> x75[x79_0]))
in (let x77 = (imap1 27 (\x80_0 -> (x75[x80_0] F.* (one F./ x76))))
in ((if ((x74_2 == x74_1)) then ((F.neg x62[x74_0]) F.* target[x74_0][x74_1]) else zero) F.* (one F./ x77[x74_2])))))))
in (let x68 = (imap2 16 27 (\x81_0 x81_1 -> (isum1 27 (\x82_0 -> (F.neg ((x67[x81_0][x81_1][x82_0] F.* x65[x81_0][x81_1][x82_0]) F.* (one F./ (x66[x81_0][x81_1] F.* x66[x81_0][x81_1]))))))))
in (let x69 = (imap3 16 27 27 (\x83_0 x83_1 x83_2 -> ((x67[x83_0][x83_1][x83_2] F.* (one F./ x66[x83_0][x83_1])) F.+ x68[x83_0][x83_1])))
in (imap2 16 27 (\x70_0 x70_1 -> (isum1 27 (\x84_0 -> ((F.exp x59[x70_0][x70_1]) F.* x69[x70_0][x84_0][x70_1]))))))))))
let x85 = (imap2 16 16 (\x86_0 x86_1 -> (isum1 27 (\x87_0 -> (isum1 16 (\x88_0 -> ((if ((x86_1 == x88_0)) then x64[x86_0][x87_0] else zero) F.* wvoc[x87_0][x86_1])))))))
let x89 = (imap2 16 64 (\x90_0 x90_1 -> (indicatorp x51[x90_0][x90_1] F.* (isum1 16 (\x91_0 -> (isum1 64 (\x92_0 -> ((if ((x90_1 == x92_0)) then x85[x90_0][x91_0] else zero) F.* wdown[x91_0][x90_1]))))))))
let x93 = (let x94 = (imap1 16 (\x102_0 -> ((isum1 16 (\x103_0 -> (x41[x102_0][x103_0] F.* x41[x102_0][x103_0]))) F./ fromi64 16)))
in (let x95 = (imap1 16 (\x104_0 -> (F.sqrt (x94[x104_0] F.+ (one F./ fromi64 100000)))))
in (let x96 = (imap2 16 16 (\x105_0 x105_1 -> (isum1 64 (\x106_0 -> (isum1 16 (\x107_0 -> ((if ((x105_1 == x107_0)) then x89[x105_0][x106_0] else zero) F.* wup[x106_0][x105_1])))))))
in (let x97 = (imap1 16 (\x108_0 -> (isum1 16 (\x109_0 -> (F.neg ((x96[x108_0][x109_0] F.* x41[x108_0][x109_0]) F.* (one F./ (x95[x108_0] F.* x95[x108_0]))))))))
in (let x98 = (imap1 16 (\x110_0 -> (x97[x110_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x94[x110_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x101_0 x101_1 -> (x85[x101_0][x101_1] F.+ (isum1 16 (\x99_0 -> ((if ((x101_0 == x99_0)) then (x96[x99_0][x101_1] F.* (one F./ x95[x99_0])) else zero) F.+ (isum1 16 (\x100_0 -> ((if ((x101_0 == x99_0)) then ((if ((x101_1 == x100_0)) then (x98[x99_0] F./ fromi64 16) else zero) F.* x41[x99_0][x101_1]) else zero) F.+ (if ((x101_0 == x99_0)) then ((if ((x101_1 == x100_0)) then (x98[x99_0] F./ fromi64 16) else zero) F.* x41[x99_0][x101_1]) else zero)))))))))))))))
let x111 = (imap3 4 16 4 (\x112_0 x112_1 x112_2 -> (isum1 16 (\x113_0 -> (isum1 16 (\x114_0 -> ((if ((((x112_0 * 4) + x112_2) == x114_0)) then x93[x112_1][x113_0] else zero) F.* wout[x113_0][((x112_0 * 4) + x112_2)])))))))
let x115 = (imap2 16 16 (\x116_0 x116_1 -> (let x117 = (imap3 4 16 16 (\x121_0 x121_1 x121_2 -> (isum1 4 (\x122_0 -> (x14[x121_0][x121_1][x122_0] F.* x17[x121_0][x121_2][x122_0])))))
in (let x118 = (imap3 4 16 16 (\x123_0 x123_1 x123_2 -> ((x117[x123_0][x123_1][x123_2] F./ fromi64 2) F.+ mask[x123_1][x123_2])))
in (let x119 = (imap3 4 16 16 (\x124_0 x124_1 x124_2 -> (let x125 = (imap1 16 (\x128_0 -> (F.exp x118[x124_0][x124_1][x128_0])))
in (let x126 = (isum1 16 (\x129_0 -> x125[x129_0]))
in (let x127 = (imap1 16 (\x130_0 -> (x125[x130_0] F.* (one F./ x126))))
in x127[x124_2])))))
in (let x120 = (imap3 4 16 4 (\x131_0 x131_1 x131_2 -> x111[x131_0][x131_1][x131_2]))
in (isum1 16 (\x132_0 -> (x120[(x116_1 / 4)][x132_0][(x116_1 % 4)] F.* x119[(x116_1 / 4)][x132_0][x116_0])))))))))
let x133 = (imap2 16 16 (\x134_0 x134_1 -> (let x135 = (imap3 4 16 16 (\x146_0 x146_1 x146_2 -> (isum1 4 (\x147_0 -> (x14[x146_0][x146_1][x147_0] F.* x17[x146_0][x146_2][x147_0])))))
in (let x136 = (imap3 4 16 16 (\x148_0 x148_1 x148_2 -> ((x135[x148_0][x148_1][x148_2] F./ fromi64 2) F.+ mask[x148_1][x148_2])))
in (let x137 = (imap3 4 16 4 (\x149_0 x149_1 x149_2 -> x111[x149_0][x149_1][x149_2]))
in (let x138 = (imap3 4 16 16 (\x150_0 x150_1 x150_2 -> (isum1 4 (\x151_0 -> (x137[x150_0][x150_1][x151_0] F.* x20[x150_0][x150_2][x151_0])))))
in (let x139 = (imap3 4 16 16 (\x152_0 x152_1 x152_2 -> (F.exp x136[x152_0][x152_1][x152_2])))
in (let x140 = (imap2 4 16 (\x153_0 x153_1 -> (isum1 16 (\x154_0 -> x139[x153_0][x153_1][x154_0]))))
in (let x141 = (imap3 4 16 16 (\x155_0 x155_1 x155_2 -> x138[x155_0][x155_1][x155_2]))
in (let x142 = (imap2 4 16 (\x156_0 x156_1 -> (isum1 16 (\x157_0 -> (F.neg ((x141[x156_0][x156_1][x157_0] F.* x139[x156_0][x156_1][x157_0]) F.* (one F./ (x140[x156_0][x156_1] F.* x140[x156_0][x156_1]))))))))
in (let x143 = (imap3 4 16 16 (\x158_0 x158_1 x158_2 -> ((x141[x158_0][x158_1][x158_2] F.* (one F./ x140[x158_0][x158_1])) F.+ x142[x158_0][x158_1])))
in (let x144 = (imap3 4 16 16 (\x159_0 x159_1 x159_2 -> ((F.exp x136[x159_0][x159_1][x159_2]) F.* x143[x159_0][x159_1][x159_2])))
in (let x145 = (imap3 4 16 16 (\x160_0 x160_1 x160_2 -> (x144[x160_0][x160_1][x160_2] F./ fromi64 2)))
in (isum1 16 (\x161_0 -> (isum1 4 (\x162_0 -> ((if (((x134_1 % 4) == x162_0)) then x145[(x134_1 / 4)][x161_0][x134_0] else zero) F.* x14[(x134_1 / 4)][x161_0][(x134_1 % 4)]))))))))))))))))))
let x163 = (imap2 16 16 (\x164_0 x164_1 -> (let x165 = (imap3 4 16 16 (\x176_0 x176_1 x176_2 -> (isum1 4 (\x177_0 -> (x14[x176_0][x176_1][x177_0] F.* x17[x176_0][x176_2][x177_0])))))
in (let x166 = (imap3 4 16 16 (\x178_0 x178_1 x178_2 -> ((x165[x178_0][x178_1][x178_2] F./ fromi64 2) F.+ mask[x178_1][x178_2])))
in (let x167 = (imap3 4 16 4 (\x179_0 x179_1 x179_2 -> x111[x179_0][x179_1][x179_2]))
in (let x168 = (imap3 4 16 16 (\x180_0 x180_1 x180_2 -> (isum1 4 (\x181_0 -> (x167[x180_0][x180_1][x181_0] F.* x20[x180_0][x180_2][x181_0])))))
in (let x169 = (imap3 4 16 16 (\x182_0 x182_1 x182_2 -> (F.exp x166[x182_0][x182_1][x182_2])))
in (let x170 = (imap2 4 16 (\x183_0 x183_1 -> (isum1 16 (\x184_0 -> x169[x183_0][x183_1][x184_0]))))
in (let x171 = (imap3 4 16 16 (\x185_0 x185_1 x185_2 -> x168[x185_0][x185_1][x185_2]))
in (let x172 = (imap2 4 16 (\x186_0 x186_1 -> (isum1 16 (\x187_0 -> (F.neg ((x171[x186_0][x186_1][x187_0] F.* x169[x186_0][x186_1][x187_0]) F.* (one F./ (x170[x186_0][x186_1] F.* x170[x186_0][x186_1]))))))))
in (let x173 = (imap3 4 16 16 (\x188_0 x188_1 x188_2 -> ((x171[x188_0][x188_1][x188_2] F.* (one F./ x170[x188_0][x188_1])) F.+ x172[x188_0][x188_1])))
in (let x174 = (imap3 4 16 16 (\x189_0 x189_1 x189_2 -> ((F.exp x166[x189_0][x189_1][x189_2]) F.* x173[x189_0][x189_1][x189_2])))
in (let x175 = (imap3 4 16 16 (\x190_0 x190_1 x190_2 -> (x174[x190_0][x190_1][x190_2] F./ fromi64 2)))
in (isum1 16 (\x191_0 -> (isum1 4 (\x192_0 -> ((if (((x164_1 % 4) == x192_0)) then x175[(x164_1 / 4)][x164_0][x191_0] else zero) F.* x17[(x164_1 / 4)][x191_0][(x164_1 % 4)]))))))))))))))))))
let x193 = (let x194 = (imap1 16 (\x202_0 -> ((isum1 16 (\x203_0 -> (x0[x202_0][x203_0] F.* x0[x202_0][x203_0]))) F./ fromi64 16)))
in (let x195 = (imap1 16 (\x204_0 -> (F.sqrt (x194[x204_0] F.+ (one F./ fromi64 100000)))))
in (let x196 = (imap2 16 16 (\x205_0 x205_1 -> (((isum1 16 (\x206_0 -> (isum1 16 (\x207_0 -> ((if ((x205_1 == x207_0)) then x115[x205_0][x206_0] else zero) F.* wval[x206_0][x205_1]))))) F.+ (isum1 16 (\x208_0 -> (isum1 16 (\x209_0 -> ((if ((x205_1 == x209_0)) then x133[x205_0][x208_0] else zero) F.* wkey[x208_0][x205_1])))))) F.+ (isum1 16 (\x210_0 -> (isum1 16 (\x211_0 -> ((if ((x205_1 == x211_0)) then x163[x205_0][x210_0] else zero) F.* wqry[x210_0][x205_1]))))))))
in (let x197 = (imap1 16 (\x212_0 -> (isum1 16 (\x213_0 -> (F.neg ((x196[x212_0][x213_0] F.* x0[x212_0][x213_0]) F.* (one F./ (x195[x212_0] F.* x195[x212_0]))))))))
in (let x198 = (imap1 16 (\x214_0 -> (x197[x214_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x194[x214_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x201_0 x201_1 -> (x93[x201_0][x201_1] F.+ (isum1 16 (\x199_0 -> ((if ((x201_0 == x199_0)) then (x196[x199_0][x201_1] F.* (one F./ x195[x199_0])) else zero) F.+ (isum1 16 (\x200_0 -> ((if ((x201_0 == x199_0)) then ((if ((x201_1 == x200_0)) then (x198[x199_0] F./ fromi64 16) else zero) F.* x0[x199_0][x201_1]) else zero) F.+ (if ((x201_0 == x199_0)) then ((if ((x201_1 == x200_0)) then (x198[x199_0] F./ fromi64 16) else zero) F.* x0[x199_0][x201_1]) else zero)))))))))))))))

let dmask = (let x215 = (imap3 4 16 16 (\x227_0 x227_1 x227_2 -> (isum1 4 (\x228_0 -> (x14[x227_0][x227_1][x228_0] F.* x17[x227_0][x227_2][x228_0])))))
in (let x216 = (imap3 4 16 16 (\x229_0 x229_1 x229_2 -> ((x215[x229_0][x229_1][x229_2] F./ fromi64 2) F.+ mask[x229_1][x229_2])))
in (let x217 = (imap3 4 16 4 (\x230_0 x230_1 x230_2 -> x111[x230_0][x230_1][x230_2]))
in (let x218 = (imap3 4 16 16 (\x231_0 x231_1 x231_2 -> (isum1 4 (\x232_0 -> (x217[x231_0][x231_1][x232_0] F.* x20[x231_0][x231_2][x232_0])))))
in (let x219 = (imap3 4 16 16 (\x233_0 x233_1 x233_2 -> (F.exp x216[x233_0][x233_1][x233_2])))
in (let x220 = (imap2 4 16 (\x234_0 x234_1 -> (isum1 16 (\x235_0 -> x219[x234_0][x234_1][x235_0]))))
in (let x221 = (imap3 4 16 16 (\x236_0 x236_1 x236_2 -> x218[x236_0][x236_1][x236_2]))
in (let x222 = (imap2 4 16 (\x237_0 x237_1 -> (isum1 16 (\x238_0 -> (F.neg ((x221[x237_0][x237_1][x238_0] F.* x219[x237_0][x237_1][x238_0]) F.* (one F./ (x220[x237_0][x237_1] F.* x220[x237_0][x237_1]))))))))
in (let x223 = (imap3 4 16 16 (\x239_0 x239_1 x239_2 -> ((x221[x239_0][x239_1][x239_2] F.* (one F./ x220[x239_0][x239_1])) F.+ x222[x239_0][x239_1])))
in (let x224 = (imap3 4 16 16 (\x240_0 x240_1 x240_2 -> ((F.exp x216[x240_0][x240_1][x240_2]) F.* x223[x240_0][x240_1][x240_2])))
in (imap2 16 16 (\x226_0 x226_1 -> (isum1 4 (\x225_0 -> x224[x225_0][x226_0][x226_1]))))))))))))))
let dwpe = (let x241 = (imap1 16 (\x249_0 -> ((isum1 16 (\x250_0 -> ((wpe[x249_0][x250_0] F.+ wseq[x249_0][x250_0]) F.* (wpe[x249_0][x250_0] F.+ wseq[x249_0][x250_0])))) F./ fromi64 16)))
in (let x242 = (imap1 16 (\x251_0 -> (F.sqrt (x241[x251_0] F.+ (one F./ fromi64 100000)))))
in (let x243 = (imap2 16 16 (\x252_0 x252_1 -> x193[x252_0][x252_1]))
in (let x244 = (imap1 16 (\x253_0 -> (isum1 16 (\x254_0 -> (F.neg ((x243[x253_0][x254_0] F.* (wpe[x253_0][x254_0] F.+ wseq[x253_0][x254_0])) F.* (one F./ (x242[x253_0] F.* x242[x253_0]))))))))
in (let x245 = (imap1 16 (\x255_0 -> (x244[x255_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x241[x255_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x248_0 x248_1 -> (isum1 16 (\x246_0 -> ((if ((x248_0 == x246_0)) then (x243[x246_0][x248_1] F.* (one F./ x242[x246_0])) else zero) F.+ (isum1 16 (\x247_0 -> ((if ((x248_0 == x246_0)) then ((if ((x248_1 == x247_0)) then (x245[x246_0] F./ fromi64 16) else zero) F.* (wpe[x246_0][x248_1] F.+ wseq[x246_0][x248_1])) else zero) F.+ (if ((x248_0 == x246_0)) then ((if ((x248_1 == x247_0)) then (x245[x246_0] F./ fromi64 16) else zero) F.* (wpe[x246_0][x248_1] F.+ wseq[x246_0][x248_1])) else zero))))))))))))))
let dwqry = (imap2 16 16 (\x256_0 x256_1 -> (isum1 16 (\x257_0 -> (isum1 16 (\x258_0 -> ((if ((x256_1 == x258_0)) then x163[x257_0][x256_0] else zero) F.* x7[x257_0][x256_1])))))))
let dwkey = (imap2 16 16 (\x259_0 x259_1 -> (isum1 16 (\x260_0 -> (isum1 16 (\x261_0 -> ((if ((x259_1 == x261_0)) then x133[x260_0][x259_0] else zero) F.* x7[x260_0][x259_1])))))))
let dwval = (imap2 16 16 (\x262_0 x262_1 -> (isum1 16 (\x263_0 -> (isum1 16 (\x264_0 -> ((if ((x262_1 == x264_0)) then x115[x263_0][x262_0] else zero) F.* x7[x263_0][x262_1])))))))
let dwout = (imap2 16 16 (\x265_0 x265_1 -> (isum1 16 (\x266_0 -> (isum1 16 (\x267_0 -> ((if ((x265_1 == x267_0)) then x93[x266_0][x265_0] else zero) F.* x23[x266_0][x265_1])))))))
let dwup = (imap2 64 16 (\x268_0 x268_1 -> (isum1 16 (\x269_0 -> (isum1 16 (\x270_0 -> ((if ((x268_1 == x270_0)) then x89[x269_0][x268_0] else zero) F.* x44[x269_0][x268_1])))))))
let dwdown = (imap2 16 64 (\x271_0 x271_1 -> (isum1 16 (\x272_0 -> (isum1 64 (\x273_0 -> ((if ((x271_1 == x273_0)) then x85[x272_0][x271_0] else zero) F.* x54[x272_0][x271_1])))))))
let dwvoc = (imap2 27 16 (\x274_0 x274_1 -> (isum1 16 (\x275_0 -> (isum1 16 (\x276_0 -> ((if ((x274_1 == x276_0)) then x64[x275_0][x274_0] else zero) F.* x56[x275_0][x274_1])))))))
let dwseq = (let x277 = (imap1 16 (\x285_0 -> ((isum1 16 (\x286_0 -> ((wpe[x285_0][x286_0] F.+ wseq[x285_0][x286_0]) F.* (wpe[x285_0][x286_0] F.+ wseq[x285_0][x286_0])))) F./ fromi64 16)))
in (let x278 = (imap1 16 (\x287_0 -> (F.sqrt (x277[x287_0] F.+ (one F./ fromi64 100000)))))
in (let x279 = (imap2 16 16 (\x288_0 x288_1 -> x193[x288_0][x288_1]))
in (let x280 = (imap1 16 (\x289_0 -> (isum1 16 (\x290_0 -> (F.neg ((x279[x289_0][x290_0] F.* (wpe[x289_0][x290_0] F.+ wseq[x289_0][x290_0])) F.* (one F./ (x278[x289_0] F.* x278[x289_0]))))))))
in (let x281 = (imap1 16 (\x291_0 -> (x280[x291_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x277[x291_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x284_0 x284_1 -> (isum1 16 (\x282_0 -> ((if ((x284_0 == x282_0)) then (x279[x282_0][x284_1] F.* (one F./ x278[x282_0])) else zero) F.+ (isum1 16 (\x283_0 -> ((if ((x284_0 == x282_0)) then ((if ((x284_1 == x283_0)) then (x281[x282_0] F./ fromi64 16) else zero) F.* (wpe[x282_0][x284_1] F.+ wseq[x282_0][x284_1])) else zero) F.+ (if ((x284_0 == x282_0)) then ((if ((x284_1 == x283_0)) then (x281[x282_0] F./ fromi64 16) else zero) F.* (wpe[x282_0][x284_1] F.+ wseq[x282_0][x284_1])) else zero))))))))))))))
let dtarget = (imap2 16 27 (\x292_0 x292_1 -> (let x293 = (imap1 27 (\x296_0 -> (F.exp x59[x292_0][x296_0])))
in (let x294 = (isum1 27 (\x297_0 -> x293[x297_0]))
in (let x295 = (imap1 27 (\x298_0 -> (x293[x298_0] F.* (one F./ x294))))
in ((F.neg x62[x292_0]) F.* (F.log x295[x292_1])))))))

in (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq)
-- in (x46, x50, wkey, wval, wout, wup, wdown, wvoc, wseq)
}

module nn64 = nn f64

type params [sl] = {
  wte:   [27][sl]f64, -- token embeddings
  wpe:   [sl][16]f64, -- position embeddings
  wqry:  [16][16]f64, -- query weights
  wkey:  [16][16]f64, -- key weights
  wval:  [16][16]f64, -- value weights
  wout:  [16][16]f64, -- output weights
  wup:   [64][16]f64, -- MLP up-projection
  wdown: [16][64]f64, -- MLP down-projection
  wvoc:  [27][16]f64  -- output projection
}

entry make_params [sl] (wte: [27][sl]f64)  (wpe: [sl][16]f64)
    (wqry: [16][16]f64) (wkey: [16][16]f64) (wval: [16][16]f64)
    (wout: [16][16]f64) (wup: [64][16]f64) (wdown: [16][64]f64)
    (wvoc: [27][16]f64) : params [sl] =
    {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc}

entry forward_seq (p : params [16]) (seq_ids : [16]i64) (mask : [16][16]f64) : [16][27]f64 =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let wseq = (imap2 16 16 (\m n -> wte[seq_ids[m]][n]))
   in nn64.forward_seq mask wpe wqry wkey wval wout wup wdown wvoc wseq

entry cal_loss (p : params [16]) (seq_ids : [16]i64) (target : [16][27]f64) (mask : [16][16]f64) : (f64 , [16]f64) =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let wseq = (imap2 16 16 (\m n -> wte[seq_ids[m]][n]))
   in nn64.cal_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target

-- entry cal_loss (asl : i64) (p : params [16]) (seq_ids : [16]i64) (mask : [16][16]f64) : (f64 , [16]f64) =
--    let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
--    let wseq = (imap2 16 16 (\m n -> wte[seq_ids[m]][n]))
--    let target_ids = (imap1 (asl - 1) (\m -> seq_ids[m + 1]))
--    -- inefficient?
--    let target = nn64.cal_target target_ids
--    in nn64.cal_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target

entry grad_loss (p : params [16]) (seq_ids : [16]i64) (target : [16][27]f64) (mask : [16][16]f64) :
        (
        [16][16]f64, -- dwpe
        [16][16]f64, -- dwqry
        [16][16]f64, -- dwkey
        [16][16]f64, -- dwval
        [16][16]f64, -- dwout
        [64][16]f64, -- dwup
        [16][64]f64, -- dwdown
        [27][16]f64, -- dwvoc
        [16][16]f64 -- dwseq
        ) =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let wseq = (imap2 16 16 (\m n -> wte[seq_ids[m]][n]))
   in nn64.grad_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target

-- entry grad_loss (asl : i64) (p : params [16]) (seq_ids : [16]i64) (mask : [16][16]f64) :
--         (
--         [16][16]f64, -- dwpe
--         [16][16]f64, -- dwqry
--         [16][16]f64, -- dwkey
--         [16][16]f64, -- dwval
--         [16][16]f64, -- dwout
--         [64][16]f64, -- dwup
--         [16][64]f64, -- dwdown
--         [27][16]f64, -- dwvoc
--         [16][16]f64 -- dwseq
--         ) =
--    let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
--    let wseq = (imap2 16 16 (\m n -> wte[seq_ids[m]][n]))
--    let target_ids = (imap1 (asl - 1) (\m -> seq_ids[m + 1]))
--    -- inefficient?
--    let target = nn64.cal_target target_ids
--   --  in (wpe, wpe, wpe, wpe, wpe, wup, wdown, wvoc, wseq)
--    in nn64.grad_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target

