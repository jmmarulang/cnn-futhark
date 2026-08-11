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

  def imaximum3 : (m: i64)
  -> (n: i64)
  -> (k: i64)
  -> (i64 -> i64 -> i64 -> real) -> real =
    \n m k f -> F.maximum (imap1 n (\i -> imaximum2 m k (f i)))

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
in (let x43 = (imap2 16 16 (\x48_0 x48_1 -> (let x49 = (imaximum1 16 (\x53_0 -> x42[x48_0][x53_0]))
in (let x50 = (imap1 16 (\x54_0 -> (F.exp (x42[x48_0][x54_0] F.+ (F.neg x49)))))
in (let x51 = (isum1 16 (\x55_0 -> x50[x55_0]))
in (let x52 = (imap1 16 (\x56_0 -> (x50[x56_0] F.* (one F./ x51))))
in x52[x48_1]))))))
in (let x44 = (imap2 16 4 (\x57_0 x57_1 -> (isum1 16 (\x58_0 -> (x43[x57_0][x58_0] F.* x7[x40_0][x58_0][x57_1])))))
in x44[x40_1][x40_2]))))))
in (let x9 = (imap2 16 16 (\x59_0 x59_1 -> x8[(x59_1 / 4)][x59_0][(x59_1 % 4)]))
in (let x10 = (imap2 16 16 (\x60_0 x60_1 -> (isum1 16 (\x61_0 -> (wout[x60_1][x61_0] F.* x9[x60_0][x61_0])))))
in (let x11 = (imap2 16 16 (\x62_0 x62_1 -> (x10[x62_0][x62_1] F.+ x0[x62_0][x62_1])))
in (let x12 = (imap2 16 16 (\x63_0 x63_1 -> (let x64 = ((isum1 16 (\x67_0 -> (x11[x63_0][x67_0] F.* x11[x63_0][x67_0]))) F./ fromi64 16)
in (let x65 = (F.sqrt (x64 F.+ (one F./ fromi64 100000)))
in (let x66 = (imap1 16 (\x68_0 -> (x11[x63_0][x68_0] F.* (one F./ x65))))
in x66[x63_1])))))
in (let x13 = (imap2 16 64 (\x69_0 x69_1 -> (isum1 16 (\x70_0 -> (wup[x69_1][x70_0] F.* x12[x69_0][x70_0])))))
in (let x14 = (imap2 16 64 (\x71_0 x71_1 -> F.max x13[x71_0][x71_1] zero))
in (let x15 = (imap2 16 16 (\x72_0 x72_1 -> (isum1 64 (\x73_0 -> (wdown[x72_1][x73_0] F.* x14[x72_0][x73_0])))))
in (let x16 = (imap2 16 16 (\x74_0 x74_1 -> (x15[x74_0][x74_1] F.+ x11[x74_0][x74_1])))
in (let x17 = (imap2 16 27 (\x75_0 x75_1 -> (isum1 16 (\x76_0 -> (wvoc[x75_1][x76_0] F.* x16[x75_0][x76_0])))))
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
in (let x8 = (imap3 4 16 4 (\x41_0 x41_1 x41_2 -> (let x42 = (imap2 16 16 (\x46_0 x46_1 -> (isum1 4 (\x47_0 -> (x5[x41_0][x46_0][x47_0] F.* x6[x41_0][x46_1][x47_0])))))
in (let x43 = (imap2 16 16 (\x48_0 x48_1 -> ((x42[x48_0][x48_1] F./ fromi64 2) F.+ mask[x48_0][x48_1])))
in (let x44 = (imap2 16 16 (\x49_0 x49_1 -> (let x50 = (imaximum1 16 (\x54_0 -> x43[x49_0][x54_0]))
in (let x51 = (imap1 16 (\x55_0 -> (F.exp (x43[x49_0][x55_0] F.+ (F.neg x50)))))
in (let x52 = (isum1 16 (\x56_0 -> x51[x56_0]))
in (let x53 = (imap1 16 (\x57_0 -> (x51[x57_0] F.* (one F./ x52))))
in x53[x49_1]))))))
in (let x45 = (imap2 16 4 (\x58_0 x58_1 -> (isum1 16 (\x59_0 -> (x44[x58_0][x59_0] F.* x7[x41_0][x59_0][x58_1])))))
in x45[x41_1][x41_2]))))))
in (let x9 = (imap2 16 16 (\x60_0 x60_1 -> x8[(x60_1 / 4)][x60_0][(x60_1 % 4)]))
in (let x10 = (imap2 16 16 (\x61_0 x61_1 -> (isum1 16 (\x62_0 -> (wout[x61_1][x62_0] F.* x9[x61_0][x62_0])))))
in (let x11 = (imap2 16 16 (\x63_0 x63_1 -> (x10[x63_0][x63_1] F.+ x0[x63_0][x63_1])))
in (let x12 = (imap2 16 16 (\x64_0 x64_1 -> (let x65 = ((isum1 16 (\x68_0 -> (x11[x64_0][x68_0] F.* x11[x64_0][x68_0]))) F./ fromi64 16)
in (let x66 = (F.sqrt (x65 F.+ (one F./ fromi64 100000)))
in (let x67 = (imap1 16 (\x69_0 -> (x11[x64_0][x69_0] F.* (one F./ x66))))
in x67[x64_1])))))
in (let x13 = (imap2 16 64 (\x70_0 x70_1 -> (isum1 16 (\x71_0 -> (wup[x70_1][x71_0] F.* x12[x70_0][x71_0])))))
in (let x14 = (imap2 16 64 (\x72_0 x72_1 -> F.max x13[x72_0][x72_1] zero))
in (let x15 = (imap2 16 16 (\x73_0 x73_1 -> (isum1 64 (\x74_0 -> (wdown[x73_1][x74_0] F.* x14[x73_0][x74_0])))))
in (let x16 = (imap2 16 16 (\x75_0 x75_1 -> (x15[x75_0][x75_1] F.+ x11[x75_0][x75_1])))
in (let x17 = (imap2 16 27 (\x76_0 x76_1 -> (isum1 16 (\x77_0 -> (wvoc[x76_1][x77_0] F.* x16[x76_0][x77_0])))))
in (let x18 = (imap1 16 (\x78_0 -> (F.neg (isum1 27 (\x79_0 -> (let x80 = (imaximum1 27 (\x84_0 -> x17[x78_0][x84_0]))
in (let x81 = (imap1 27 (\x85_0 -> (F.exp (x17[x78_0][x85_0] F.+ (F.neg x80)))))
in (let x82 = (isum1 27 (\x86_0 -> x81[x86_0]))
in (let x83 = (imap1 27 (\x87_0 -> (x81[x87_0] F.* (one F./ x82))))
in ((F.log x83[x79_0]) F.* target[x78_0][x79_0]))))))))))
in (let x19 = ((isum1 16 (\x88_0 -> x18[x88_0])) F./ fromi64 16)
in
--x19
let loss = x19
let losses = x18
in (loss, losses)
))))))))))))))))))))

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
let x14 = (imap2 16 16 (\x15_0 x15_1 -> (isum1 16 (\x16_0 -> (wqry[x15_1][x16_0] F.* x7[x15_0][x16_0])))))
let x17 = (imap2 16 16 (\x18_0 x18_1 -> (isum1 16 (\x19_0 -> (wkey[x18_1][x19_0] F.* x7[x18_0][x19_0])))))
let x20 = (imap2 16 16 (\x21_0 x21_1 -> (isum1 16 (\x22_0 -> (wval[x21_1][x22_0] F.* x7[x21_0][x22_0])))))
let x23 = (imap3 4 16 4 (\x24_0 x24_1 x24_2 -> x14[x24_1][((x24_0 * 4) + x24_2)]))
let x25 = (imap3 4 16 4 (\x26_0 x26_1 x26_2 -> x17[x26_1][((x26_0 * 4) + x26_2)]))
let x27 = (imap3 4 16 4 (\x28_0 x28_1 x28_2 -> x20[x28_1][((x28_0 * 4) + x28_2)]))
let x29 = (imap3 4 16 4 (\x30_0 x30_1 x30_2 -> (let x31 = (imap2 16 16 (\x35_0 x35_1 -> (isum1 4 (\x36_0 -> (x23[x30_0][x35_0][x36_0] F.* x25[x30_0][x35_1][x36_0])))))
in (let x32 = (imap2 16 16 (\x37_0 x37_1 -> ((x31[x37_0][x37_1] F./ fromi64 2) F.+ mask[x37_0][x37_1])))
in (let x33 = (imap2 16 16 (\x38_0 x38_1 -> (let x39 = (imaximum1 16 (\x43_0 -> x32[x38_0][x43_0]))
in (let x40 = (imap1 16 (\x44_0 -> (F.exp (x32[x38_0][x44_0] F.+ (F.neg x39)))))
in (let x41 = (isum1 16 (\x45_0 -> x40[x45_0]))
in (let x42 = (imap1 16 (\x46_0 -> (x40[x46_0] F.* (one F./ x41))))
in x42[x38_1]))))))
in (let x34 = (imap2 16 4 (\x47_0 x47_1 -> (isum1 16 (\x48_0 -> (x33[x47_0][x48_0] F.* x27[x30_0][x48_0][x47_1])))))
in x34[x30_1][x30_2]))))))
let x49 = (imap2 16 16 (\x50_0 x50_1 -> x29[(x50_1 / 4)][x50_0][(x50_1 % 4)]))
let x51 = (imap2 16 16 (\x52_0 x52_1 -> (isum1 16 (\x53_0 -> (wout[x52_1][x53_0] F.* x49[x52_0][x53_0])))))
let x54 = (imap2 16 16 (\x55_0 x55_1 -> (x51[x55_0][x55_1] F.+ x0[x55_0][x55_1])))
let x56 = (imap2 16 16 (\x57_0 x57_1 -> (let x58 = ((isum1 16 (\x61_0 -> (x54[x57_0][x61_0] F.* x54[x57_0][x61_0]))) F./ fromi64 16)
in (let x59 = (F.sqrt (x58 F.+ (one F./ fromi64 100000)))
in (let x60 = (imap1 16 (\x62_0 -> (x54[x57_0][x62_0] F.* (one F./ x59))))
in x60[x57_1])))))
let x63 = (imap2 16 64 (\x64_0 x64_1 -> (isum1 16 (\x65_0 -> (wup[x64_1][x65_0] F.* x56[x64_0][x65_0])))))
let x66 = (imap2 16 64 (\x67_0 x67_1 -> F.max x63[x67_0][x67_1] zero))
let x68 = (imap2 16 16 (\x69_0 x69_1 -> (isum1 64 (\x70_0 -> (wdown[x69_1][x70_0] F.* x66[x69_0][x70_0])))))
let x71 = (imap2 16 16 (\x72_0 x72_1 -> (x68[x72_0][x72_1] F.+ x54[x72_0][x72_1])))
let x73 = (imap2 16 27 (\x74_0 x74_1 -> (isum1 16 (\x75_0 -> (wvoc[x74_1][x75_0] F.* x71[x74_0][x75_0])))))
let x76 = one
let x77 = (imap1 16 (\x78_0 -> (x76 F./ fromi64 16)))
let x79 = (imap1 16 (\x80_0 -> (imaximum1 27 (\x81_0 -> x73[x80_0][x81_0]))))
let x82 = (imap2 16 27 (\x83_0 x83_1 -> (F.exp (x73[x83_0][x83_1] F.+ (F.neg x79[x83_0])))))
let x84 = (imap1 16 (\x85_0 -> (isum1 27 (\x86_0 -> x82[x85_0][x86_0]))))
let x87 = (imap3 16 27 27 (\x88_0 x88_1 x88_2 -> (if ((x88_2 == x88_1)) then (let x89 = (imaximum1 27 (\x93_0 -> x73[x88_0][x93_0]))
in (let x90 = (imap1 27 (\x94_0 -> (F.exp (x73[x88_0][x94_0] F.+ (F.neg x89)))))
in (let x91 = (isum1 27 (\x95_0 -> x90[x95_0]))
in (let x92 = (imap1 27 (\x96_0 -> (x90[x96_0] F.* (one F./ x91))))
in (((F.neg x77[x88_0]) F.* target[x88_0][x88_1]) F.* (one F./ x92[x88_2])))))) else zero)))
let x97 = (imap2 16 27 (\x98_0 x98_1 -> (isum1 27 (\x99_0 -> (F.neg ((x87[x98_0][x98_1][x99_0] F.* x82[x98_0][x99_0]) F.* (one F./ (x84[x98_0] F.* x84[x98_0]))))))))
let x100 = (imap3 16 27 27 (\x101_0 x101_1 x101_2 -> ((x87[x101_0][x101_1][x101_2] F.* (one F./ x84[x101_0])) F.+ x97[x101_0][x101_1])))
let x102 = (imap2 16 27 (\x103_0 x103_1 -> (isum1 27 (\x104_0 -> (F.neg ((F.exp (x73[x103_0][x104_0] F.+ (F.neg x79[x103_0]))) F.* x100[x103_0][x103_1][x104_0]))))))
let x105 = (imap1 16 (\x106_0 -> x79[x106_0]))
let x107 = (imap1 16 (\x108_0 -> (one F./ (isum1 27 (\x109_0 -> (x76 F.+ (F.neg (indicatorp (F.neg (x73[x108_0][x109_0] F.+ (F.neg x105[x108_0])))))))))))
let x110 = (imap2 16 27 (\x111_0 x111_1 -> (isum1 27 (\x112_0 -> (((F.exp (x73[x111_0][x111_1] F.+ (F.neg x79[x111_0]))) F.* x100[x111_0][x112_0][x111_1]) F.+ ((x102[x111_0][x112_0] F.* (x76 F.+ (F.neg (indicatorp (F.neg (x73[x111_0][x111_1] F.+ (F.neg x105[x111_0]))))))) F.* x107[x111_0]))))))
let x113 = (imap2 16 16 (\x114_0 x114_1 -> (isum1 27 (\x115_0 -> (x110[x114_0][x115_0] F.* wvoc[x115_0][x114_1])))))
let x116 = (imap2 16 16 (\x117_0 x117_1 -> x113[x117_0][x117_1]))
let x118 = (imap2 16 64 (\x119_0 x119_1 -> (isum1 16 (\x120_0 -> (x116[x119_0][x120_0] F.* wdown[x120_0][x119_1])))))
let x121 = (imap2 16 64 (\x122_0 x122_1 -> ((indicatorp x63[x122_0][x122_1]) F.* x118[x122_0][x122_1])))
let x123 = (imap2 16 16 (\x124_0 x124_1 -> (isum1 64 (\x125_0 -> (x121[x124_0][x125_0] F.* wup[x125_0][x124_1])))))
let x126 = (imap1 16 (\x127_0 -> ((isum1 16 (\x128_0 -> (x54[x127_0][x128_0] F.* x54[x127_0][x128_0]))) F./ fromi64 16)))
let x129 = (imap1 16 (\x130_0 -> (F.sqrt (x126[x130_0] F.+ (x76 F./ fromi64 100000)))))
let x131 = (imap2 16 16 (\x132_0 x132_1 -> x123[x132_0][x132_1]))
let x133 = (imap1 16 (\x134_0 -> (isum1 16 (\x135_0 -> (F.neg ((x131[x134_0][x135_0] F.* x54[x134_0][x135_0]) F.* (one F./ (x129[x134_0] F.* x129[x134_0]))))))))
let x136 = (imap1 16 (\x137_0 -> (x133[x137_0] F.* (one F./ ((x76 F.+ x76) F.* (F.sqrt (x126[x137_0] F.+ (x76 F./ fromi64 100000))))))))
let x138 = (imap2 16 16 (\x139_0 x139_1 -> (x116[x139_0][x139_1] F.+ ((x131[x139_0][x139_1] F.* (one F./ x129[x139_0])) F.+ (((x136[x139_0] F./ fromi64 16) F.* x54[x139_0][x139_1]) F.+ ((x136[x139_0] F./ fromi64 16) F.* x54[x139_0][x139_1]))))))
let x140 = (imap2 16 16 (\x141_0 x141_1 -> x138[x141_0][x141_1]))
let x142 = (imap2 16 16 (\x143_0 x143_1 -> (isum1 16 (\x144_0 -> (x140[x143_0][x144_0] F.* wout[x144_0][x143_1])))))
let x145 = (imap3 4 16 4 (\x146_0 x146_1 x146_2 -> x142[x146_1][((x146_0 * 4) + x146_2)]))
let x147 = (imap3 4 16 16 (\x148_0 x148_1 x148_2 -> (isum1 4 (\x149_0 -> (x23[x148_0][x148_1][x149_0] F.* x25[x148_0][x148_2][x149_0])))))
let x150 = (imap3 4 16 16 (\x151_0 x151_1 x151_2 -> ((x147[x151_0][x151_1][x151_2] F./ fromi64 2) F.+ mask[x151_1][x151_2])))
let x152 = (imap3 4 16 16 (\x153_0 x153_1 x153_2 -> (let x154 = (imaximum1 16 (\x158_0 -> x150[x153_0][x153_1][x158_0]))
in (let x155 = (imap1 16 (\x159_0 -> (F.exp (x150[x153_0][x153_1][x159_0] F.+ (F.neg x154)))))
in (let x156 = (isum1 16 (\x160_0 -> x155[x160_0]))
in (let x157 = (imap1 16 (\x161_0 -> (x155[x161_0] F.* (one F./ x156))))
in x157[x153_2]))))))
let x162 = (imap3 4 16 4 (\x163_0 x163_1 x163_2 -> x145[x163_0][x163_1][x163_2]))
let x164 = (imap3 4 16 16 (\x165_0 x165_1 x165_2 -> (isum1 4 (\x166_0 -> (x162[x165_0][x165_1][x166_0] F.* x27[x165_0][x165_2][x166_0])))))
let x167 = (imap2 4 16 (\x168_0 x168_1 -> (imaximum1 16 (\x169_0 -> x150[x168_0][x168_1][x169_0]))))
let x170 = (imap3 4 16 16 (\x171_0 x171_1 x171_2 -> (F.exp (x150[x171_0][x171_1][x171_2] F.+ (F.neg x167[x171_0][x171_1])))))
let x172 = (imap2 4 16 (\x173_0 x173_1 -> (isum1 16 (\x174_0 -> x170[x173_0][x173_1][x174_0]))))
let x175 = (imap3 4 16 16 (\x176_0 x176_1 x176_2 -> x164[x176_0][x176_1][x176_2]))
let x177 = (imap2 4 16 (\x178_0 x178_1 -> (isum1 16 (\x179_0 -> (F.neg ((x175[x178_0][x178_1][x179_0] F.* x170[x178_0][x178_1][x179_0]) F.* (one F./ (x172[x178_0][x178_1] F.* x172[x178_0][x178_1]))))))))
let x180 = (imap3 4 16 16 (\x181_0 x181_1 x181_2 -> ((x175[x181_0][x181_1][x181_2] F.* (one F./ x172[x181_0][x181_1])) F.+ x177[x181_0][x181_1])))
let x182 = (imap2 4 16 (\x183_0 x183_1 -> (isum1 16 (\x184_0 -> (F.neg ((F.exp (x150[x183_0][x183_1][x184_0] F.+ (F.neg x167[x183_0][x183_1]))) F.* x180[x183_0][x183_1][x184_0]))))))
let x185 = (imap2 4 16 (\x186_0 x186_1 -> x167[x186_0][x186_1]))
let x187 = (imap2 4 16 (\x188_0 x188_1 -> (one F./ (isum1 16 (\x189_0 -> (x76 F.+ (F.neg (indicatorp (F.neg (x150[x188_0][x188_1][x189_0] F.+ (F.neg x185[x188_0][x188_1])))))))))))
let x190 = (imap3 4 16 16 (\x191_0 x191_1 x191_2 -> (((F.exp (x150[x191_0][x191_1][x191_2] F.+ (F.neg x167[x191_0][x191_1]))) F.* x180[x191_0][x191_1][x191_2]) F.+ ((x182[x191_0][x191_1] F.* (x76 F.+ (F.neg (indicatorp (F.neg (x150[x191_0][x191_1][x191_2] F.+ (F.neg x185[x191_0][x191_1]))))))) F.* x187[x191_0][x191_1]))))
let x192 = (imap3 4 16 16 (\x193_0 x193_1 x193_2 -> (x190[x193_0][x193_1][x193_2] F./ fromi64 2)))
let x194 = (imap3 4 16 4 (\x195_0 x195_1 x195_2 -> (isum1 16 (\x196_0 -> (x162[x195_0][x196_0][x195_2] F.* x152[x195_0][x196_0][x195_1])))))
let x197 = (imap3 4 16 4 (\x198_0 x198_1 x198_2 -> (isum1 16 (\x199_0 -> (x192[x198_0][x199_0][x198_1] F.* x23[x198_0][x199_0][x198_2])))))
let x200 = (imap3 4 16 4 (\x201_0 x201_1 x201_2 -> (isum1 16 (\x202_0 -> (x192[x201_0][x201_1][x202_0] F.* x25[x201_0][x202_0][x201_2])))))
let x203 = (imap2 16 16 (\x204_0 x204_1 -> x194[(x204_1 / 4)][x204_0][(x204_1 % 4)]))
let x205 = (imap2 16 16 (\x206_0 x206_1 -> x197[(x206_1 / 4)][x206_0][(x206_1 % 4)]))
let x207 = (imap2 16 16 (\x208_0 x208_1 -> x200[(x208_1 / 4)][x208_0][(x208_1 % 4)]))
let x209 = (imap2 16 16 (\x210_0 x210_1 -> (((isum1 16 (\x211_0 -> (x203[x210_0][x211_0] F.* wval[x211_0][x210_1]))) F.+ (isum1 16 (\x212_0 -> (x205[x210_0][x212_0] F.* wkey[x212_0][x210_1])))) F.+ (isum1 16 (\x213_0 -> (x207[x210_0][x213_0] F.* wqry[x213_0][x210_1]))))))
let x214 = (imap1 16 (\x215_0 -> ((isum1 16 (\x216_0 -> (x0[x215_0][x216_0] F.* x0[x215_0][x216_0]))) F./ fromi64 16)))
let x217 = (imap1 16 (\x218_0 -> (F.sqrt (x214[x218_0] F.+ (x76 F./ fromi64 100000)))))
let x219 = (imap2 16 16 (\x220_0 x220_1 -> x209[x220_0][x220_1]))
let x221 = (imap1 16 (\x222_0 -> (isum1 16 (\x223_0 -> (F.neg ((x219[x222_0][x223_0] F.* x0[x222_0][x223_0]) F.* (one F./ (x217[x222_0] F.* x217[x222_0]))))))))
let x224 = (imap1 16 (\x225_0 -> (x221[x225_0] F.* (one F./ ((x76 F.+ x76) F.* (F.sqrt (x214[x225_0] F.+ (x76 F./ fromi64 100000))))))))
let x226 = (imap2 16 16 (\x227_0 x227_1 -> (x140[x227_0][x227_1] F.+ ((x219[x227_0][x227_1] F.* (one F./ x217[x227_0])) F.+ (((x224[x227_0] F./ fromi64 16) F.* x0[x227_0][x227_1]) F.+ ((x224[x227_0] F./ fromi64 16) F.* x0[x227_0][x227_1]))))))
let x228 = (imap1 16 (\x229_0 -> ((isum1 16 (\x230_0 -> ((wpe[x229_0][x230_0] F.+ wseq[x229_0][x230_0]) F.* (wpe[x229_0][x230_0] F.+ wseq[x229_0][x230_0])))) F./ fromi64 16)))
let x231 = (imap1 16 (\x232_0 -> (F.sqrt (x228[x232_0] F.+ (x76 F./ fromi64 100000)))))
let x233 = (imap2 16 16 (\x234_0 x234_1 -> x226[x234_0][x234_1]))
let x235 = (imap1 16 (\x236_0 -> (isum1 16 (\x237_0 -> (F.neg ((x233[x236_0][x237_0] F.* (wpe[x236_0][x237_0] F.+ wseq[x236_0][x237_0])) F.* (one F./ (x231[x236_0] F.* x231[x236_0]))))))))
let x238 = (imap1 16 (\x239_0 -> (x235[x239_0] F.* (one F./ ((x76 F.+ x76) F.* (F.sqrt (x228[x239_0] F.+ (x76 F./ fromi64 100000))))))))

let dmask = (imap2 16 16 (\x241_0 x241_1 -> (isum1 4 (\x240_0 -> x190[x240_0][x241_0][x241_1]))))
let dwpe = (imap2 16 16 (\x242_0 x242_1 -> ((x233[x242_0][x242_1] F.* (one F./ x231[x242_0])) F.+ (((x238[x242_0] F./ fromi64 16) F.* (wpe[x242_0][x242_1] F.+ wseq[x242_0][x242_1])) F.+ ((x238[x242_0] F./ fromi64 16) F.* (wpe[x242_0][x242_1] F.+ wseq[x242_0][x242_1]))))))
let dwqry = (imap2 16 16 (\x243_0 x243_1 -> (isum1 16 (\x244_0 -> (x207[x244_0][x243_0] F.* x7[x244_0][x243_1])))))
let dwkey = (imap2 16 16 (\x245_0 x245_1 -> (isum1 16 (\x246_0 -> (x205[x246_0][x245_0] F.* x7[x246_0][x245_1])))))
let dwval = (imap2 16 16 (\x247_0 x247_1 -> (isum1 16 (\x248_0 -> (x203[x248_0][x247_0] F.* x7[x248_0][x247_1])))))
let dwout = (imap2 16 16 (\x249_0 x249_1 -> (isum1 16 (\x250_0 -> (x140[x250_0][x249_0] F.* x49[x250_0][x249_1])))))
let dwup = (imap2 64 16 (\x251_0 x251_1 -> (isum1 16 (\x252_0 -> (x121[x252_0][x251_0] F.* x56[x252_0][x251_1])))))
let dwdown = (imap2 16 64 (\x253_0 x253_1 -> (isum1 16 (\x254_0 -> (x116[x254_0][x253_0] F.* x66[x254_0][x253_1])))))
let dwvoc = (imap2 27 16 (\x255_0 x255_1 -> (isum1 16 (\x256_0 -> (x110[x256_0][x255_0] F.* x71[x256_0][x255_1])))))
let dwseq = (imap2 16 16 (\x257_0 x257_1 -> ((x233[x257_0][x257_1] F.* (one F./ x231[x257_0])) F.+ (((x238[x257_0] F./ fromi64 16) F.* (wpe[x257_0][x257_1] F.+ wseq[x257_0][x257_1])) F.+ ((x238[x257_0] F./ fromi64 16) F.* (wpe[x257_0][x257_1] F.+ wseq[x257_0][x257_1]))))))
let dtarget = (imap2 16 27 (\x258_0 x258_1 -> (let x259 = (imaximum1 27 (\x263_0 -> x73[x258_0][x263_0]))
in (let x260 = (imap1 27 (\x264_0 -> (F.exp (x73[x258_0][x264_0] F.+ (F.neg x259)))))
in (let x261 = (isum1 27 (\x265_0 -> x260[x265_0]))
in (let x262 = (imap1 27 (\x266_0 -> (x260[x266_0] F.* (one F./ x261))))
in ((F.neg x77[x258_0]) F.* (F.log x262[x258_1]))))))))

in (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq)
-- in (x46, x50, wkey, wval, wout, wup, wdown, wvoc, wseq)
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

entry cal_loss (p : params) (tokens : [16]i64) (target : [16][27]f64) (mask : [16][16]f64) : (f64 , [16]f64) =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let wseq = (imap2 16 16 (\m n -> wte[tokens[m]][n]))
   in nn64.cal_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target

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

-- def adam_opt (p : params) (mp : params) (vp : params)
--   (dp : params) (step : i64) (num_steps : i64) (lr : f64)
--   (beta1 : f64) (beta2 : f64) (eps_adam : f64):
--   (params,  params,  params) =
--   let lt_r = lr * (1 - (nn64.fromi64 step) / (nn64.fromi64 num_steps))
--   let (wte, mwte, vwte) =
--     adam_opt_w p.wte mp.wte vp.wte dp.wte step lt_r beta1 beta2 eps_adam
--   let (wpe, mwpe, vwpe) =
--     adam_opt_w p.wpe mp.wpe vp.wpe dp.wpe step lt_r beta1 beta2 eps_adam
--   let (wqry, mwqry, vwqry) =
--     adam_opt_w p.wqry mp.wqry vp.wqry dp.wqry step lt_r beta1 beta2 eps_adam
--   let (wkey, mwkey, vwkey) =
--     adam_opt_w p.wkey mp.wkey vp.wkey dp.wkey step lt_r beta1 beta2 eps_adam
--   let (wval, mwval, vwval) =
--     adam_opt_w p.wval mp.wval vp.wval dp.wval step lt_r beta1 beta2 eps_adam
--   let (wout, mwout, vwout) =
--     adam_opt_w p.wout mp.wout vp.wout dp.wout step lt_r beta1 beta2 eps_adam
--   let (wup, mwup, vwup) =
--     adam_opt_w p.wup mp.wup vp.wup dp.wup step lt_r beta1 beta2 eps_adam
--   let (wdown, mwdown, vwdown) =
--     adam_opt_w p.wdown mp.wdown vp.wdown dp.wdown step lt_r beta1 beta2 eps_adam
--   let (wvoc, mwvoc, vwvoc) =
--     adam_opt_w p.wvoc mp.wvoc vp.wvoc dp.wvoc step lt_r beta1 beta2 eps_adam
--   let p' = to_params wte wpe wqry wkey wval wout wup wdown wvoc
--   let mp' = to_params mwte mwpe mwqry mwkey mwval mwout mwup mwdown mwvoc
--   let vp' = to_params vwte vwpe vwqry vwkey vwval vwout vwup vwdown vwvoc
--   in (p', mp', vp')

def adam_opt (p : params) (mp : params) (vp : params)
  (dp : params) (step : i64):
  (params,  params,  params) =
  let lt_r = 0.01 * (1 - (nn64.fromi64 step) / (nn64.fromi64 11))
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

-- entry grad_loss (p : params) (tokens : [16]i64) (target : [16][27]f64) (mask : [16][16]f64) :
-- def grad_loss (n : i64) (p : params) (tokens : [16]i64) (mask : [16][16]f64) :
--         (
--         [27][16]f64, -- dwte
--         [16][16]f64, -- dwpe
--         [16][16]f64, -- dwqry
--         [16][16]f64, -- dwkey
--         [16][16]f64, -- dwval
--         [16][16]f64, -- dwout
--         [64][16]f64, -- dwup
--         [16][64]f64, -- dwdown
--         [27][16]f64, -- dwvoc
--         ) =
--    let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
--    let target = cal_target n tokens
--    let wseq = (imap2 16 16 (\m n -> wte[tokens[m]][n]))
--    let (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq) =
--     nn64.grad_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target
--    let dwte = (imap2 27 16 (\m n -> nn64.isum1 16 (\k -> if (tokens[k] == m) then dwseq[k][n] else nn64.zero)))
--    in  (dwte, dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc)

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

-- def cal_step (dl : i64) (p : params) (mp : params) (vp : params)
--   (tokens : [16]i64) (mask : [16][16]f64)
--   (step : i64) (num_steps : i64) :
--   (params,  params,  params) =
--   -- cal gradient
--   let (dwte, dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc) =
--     grad_loss dl p tokens mask
--   let dp = to_params dwte dwpe dwqry dwkey dwval dwout dwup dwdown dwvoc
--   -- cal new model weights
--   let (p', mp', vp') =
--     adam_opt p mp vp dp step num_steps 0.01 0.85 0.99 0.00000001
--   in (p', mp', vp')

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

-- entry train (num_steps : i64) (p : params) (mp : params) (vp : params)
--   (masks : [num_steps][16][16]f64) (dls : [num_steps]i64)
--   (seqs : [num_steps][16]i64) =
--   let (new_p, new_mp, new_vp) =
--     loop (p', mp', vp') = (p, mp, vp)
--     for step < num_steps do
--       let dl = dls[step]
--       let tokens = seqs[step]
--       let mask = masks[step]
--       in (cal_step dl p' mp' vp' tokens mask step num_steps)
--   in ((from_params new_p), (from_params new_mp), (from_params new_vp))

entry train (p : params) (mp : params) (vp : params)
  (masks : [11][16][16]f64) (dls : [11]i64)
  (seqs : [11][16]i64) =
  let (new_p, new_mp, new_vp) =
    loop (p', mp', vp') = (p, mp, vp)
    for step < 11 do
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