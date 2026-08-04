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
let x14 = (imap3 4 16 4 (\x15_0 x15_1 x15_2 -> (isum1 16 (\x16_0 -> (wqry[((x15_0 * 4) + x15_2)][x16_0] F.* x7[x15_1][x16_0])))))
let x17 = (imap3 4 16 4 (\x18_0 x18_1 x18_2 -> (isum1 16 (\x19_0 -> (wkey[((x18_0 * 4) + x18_2)][x19_0] F.* x7[x18_1][x19_0])))))
let x20 = (imap3 4 16 4 (\x21_0 x21_1 x21_2 -> (isum1 16 (\x22_0 -> (wval[((x21_0 * 4) + x21_2)][x22_0] F.* x7[x21_1][x22_0])))))
let x23 = (imap2 16 16 (\x24_0 x24_1 -> (let x25 = (imap2 16 16 (\x29_0 x29_1 -> (isum1 4 (\x30_0 -> (x14[(x24_1 / 4)][x29_0][x30_0] F.* x17[(x24_1 / 4)][x29_1][x30_0])))))
in (let x26 = (imap2 16 16 (\x31_0 x31_1 -> ((x25[x31_0][x31_1] F./ fromi64 2) F.+ mask[x31_0][x31_1])))
in (let x27 = (imap2 16 16 (\x32_0 x32_1 -> (let x33 = (imaximum1 16 (\x37_0 -> x26[x32_0][x37_0]))
in (let x34 = (imap1 16 (\x38_0 -> (F.exp (x26[x32_0][x38_0] F.+ (F.neg x33)))))
in (let x35 = (isum1 16 (\x39_0 -> x34[x39_0]))
in (let x36 = (imap1 16 (\x40_0 -> (x34[x40_0] F.* (one F./ x35))))
in x36[x32_1]))))))
in (let x28 = (imap2 16 4 (\x41_0 x41_1 -> (isum1 16 (\x42_0 -> (x27[x41_0][x42_0] F.* x20[(x24_1 / 4)][x42_0][x41_1])))))
in x28[x24_0][(x24_1 % 4)]))))))
let x43 = (imap2 16 16 (\x44_0 x44_1 -> ((isum1 16 (\x45_0 -> (wout[x44_1][x45_0] F.* x23[x44_0][x45_0]))) F.+ x0[x44_0][x44_1])))
let x46 = (imap2 16 16 (\x47_0 x47_1 -> (let x48 = ((isum1 16 (\x51_0 -> (x43[x47_0][x51_0] F.* x43[x47_0][x51_0]))) F./ fromi64 16)
in (let x49 = (F.sqrt (x48 F.+ (one F./ fromi64 100000)))
in (let x50 = (imap1 16 (\x52_0 -> (x43[x47_0][x52_0] F.* (one F./ x49))))
in x50[x47_1])))))
let x53 = (imap2 16 64 (\x54_0 x54_1 -> (isum1 16 (\x55_0 -> (wup[x54_1][x55_0] F.* x46[x54_0][x55_0])))))
let x56 = (imap2 16 64 (\x57_0 x57_1 -> F.max x53[x57_0][x57_1] zero))
let x58 = (imap2 16 16 (\x59_0 x59_1 -> ((isum1 64 (\x60_0 -> (wdown[x59_1][x60_0] F.* x56[x59_0][x60_0]))) F.+ x43[x59_0][x59_1])))
let x61 = (imap2 16 27 (\x62_0 x62_1 -> (isum1 16 (\x63_0 -> (wvoc[x62_1][x63_0] F.* x58[x62_0][x63_0])))))
let x64 = (imap1 16 (\x65_0 -> (one F./ fromi64 16)))
let x66 = (let x67 = (imap2 16 27 (\x77_0 x77_1 -> (imaximum1 27 (\x78_0 -> x61[x77_0][x78_0]))))
in (let x68 = (imap3 16 27 27 (\x79_0 x79_1 x79_2 -> (F.exp (x61[x79_0][x79_2] F.+ (F.neg x67[x79_0][x79_1])))))
in (let x69 = (imap2 16 27 (\x80_0 x80_1 -> (isum1 27 (\x81_0 -> x68[x80_0][x80_1][x81_0]))))
in (let x70 = (imap3 16 27 27 (\x82_0 x82_1 x82_2 -> (if ((x82_2 == x82_1)) then (let x83 = (imaximum1 27 (\x87_0 -> x61[x82_0][x87_0]))
in (let x84 = (imap1 27 (\x88_0 -> (F.exp (x61[x82_0][x88_0] F.+ (F.neg x83)))))
in (let x85 = (isum1 27 (\x89_0 -> x84[x89_0]))
in (let x86 = (imap1 27 (\x90_0 -> (x84[x90_0] F.* (one F./ x85))))
in (((F.neg x64[x82_0]) F.* target[x82_0][x82_1]) F.* (one F./ x86[x82_2])))))) else zero)))
in (let x71 = (imap2 16 27 (\x91_0 x91_1 -> (isum1 27 (\x92_0 -> (F.neg ((x70[x91_0][x91_1][x92_0] F.* x68[x91_0][x91_1][x92_0]) F.* (one F./ (x69[x91_0][x91_1] F.* x69[x91_0][x91_1]))))))))
in (let x72 = (imap3 16 27 27 (\x93_0 x93_1 x93_2 -> ((x70[x93_0][x93_1][x93_2] F.* (one F./ x69[x93_0][x93_1])) F.+ x71[x93_0][x93_1])))
in (let x73 = (imap2 16 27 (\x94_0 x94_1 -> (isum1 27 (\x95_0 -> (F.neg ((F.exp (x61[x94_0][x95_0] F.+ (F.neg x67[x94_0][x94_1]))) F.* x72[x94_0][x94_1][x95_0]))))))
in (let x74 = (imap2 16 27 (\x96_0 x96_1 -> (imaximum1 27 (\x97_0 -> x61[x96_0][x97_0]))))
in (let x75 = (imap2 16 27 (\x98_0 x98_1 -> (one F./ (isum1 27 (\x99_0 -> (one F.+ (F.neg (indicatorp (F.neg (x61[x98_0][x99_0] F.+ (F.neg x74[x98_0][x98_1])))))))))))
in (imap2 16 27 (\x76_0 x76_1 -> (isum1 27 (\x100_0 -> (((F.exp (x61[x76_0][x76_1] F.+ (F.neg x67[x76_0][x100_0]))) F.* x72[x76_0][x100_0][x76_1]) F.+ ((x73[x76_0][x100_0] F.* (one F.+ (F.neg (indicatorp (F.neg (x61[x76_0][x76_1] F.+ (F.neg x74[x76_0][x100_0]))))))) F.* x75[x76_0][x100_0])))))))))))))))
let x101 = (imap2 16 16 (\x102_0 x102_1 -> (isum1 27 (\x103_0 -> (x66[x102_0][x103_0] F.* wvoc[x103_0][x102_1])))))
let x104 = (imap2 16 64 (\x105_0 x105_1 -> ((indicatorp x53[x105_0][x105_1]) F.* (isum1 16 (\x106_0 -> (x101[x105_0][x106_0] F.* wdown[x106_0][x105_1]))))))
let x107 = (let x108 = (imap1 16 (\x114_0 -> ((isum1 16 (\x115_0 -> (x43[x114_0][x115_0] F.* x43[x114_0][x115_0]))) F./ fromi64 16)))
in (let x109 = (imap1 16 (\x116_0 -> (F.sqrt (x108[x116_0] F.+ (one F./ fromi64 100000)))))
in (let x110 = (imap2 16 16 (\x117_0 x117_1 -> (isum1 64 (\x118_0 -> (x104[x117_0][x118_0] F.* wup[x118_0][x117_1])))))
in (let x111 = (imap1 16 (\x119_0 -> (isum1 16 (\x120_0 -> (F.neg ((x110[x119_0][x120_0] F.* x43[x119_0][x120_0]) F.* (one F./ (x109[x119_0] F.* x109[x119_0]))))))))
in (let x112 = (imap1 16 (\x121_0 -> (x111[x121_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x108[x121_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x113_0 x113_1 -> (x101[x113_0][x113_1] F.+ ((x110[x113_0][x113_1] F.* (one F./ x109[x113_0])) F.+ (((x112[x113_0] F./ fromi64 16) F.* x43[x113_0][x113_1]) F.+ ((x112[x113_0] F./ fromi64 16) F.* x43[x113_0][x113_1])))))))))))
let x122 = (imap3 4 16 4 (\x123_0 x123_1 x123_2 -> (isum1 16 (\x124_0 -> (x107[x123_1][x124_0] F.* wout[x124_0][((x123_0 * 4) + x123_2)])))))
let x125 = (imap2 16 16 (\x126_0 x126_1 -> (let x127 = (imap3 4 16 16 (\x131_0 x131_1 x131_2 -> (isum1 4 (\x132_0 -> (x14[x131_0][x131_1][x132_0] F.* x17[x131_0][x131_2][x132_0])))))
in (let x128 = (imap3 4 16 16 (\x133_0 x133_1 x133_2 -> ((x127[x133_0][x133_1][x133_2] F./ fromi64 2) F.+ mask[x133_1][x133_2])))
in (let x129 = (imap3 4 16 16 (\x134_0 x134_1 x134_2 -> (let x135 = (imaximum1 16 (\x139_0 -> x128[x134_0][x134_1][x139_0]))
in (let x136 = (imap1 16 (\x140_0 -> (F.exp (x128[x134_0][x134_1][x140_0] F.+ (F.neg x135)))))
in (let x137 = (isum1 16 (\x141_0 -> x136[x141_0]))
in (let x138 = (imap1 16 (\x142_0 -> (x136[x142_0] F.* (one F./ x137))))
in x138[x134_2]))))))
in (let x130 = (imap3 4 16 4 (\x143_0 x143_1 x143_2 -> x122[x143_0][x143_1][x143_2]))
in (isum1 16 (\x144_0 -> (x130[(x126_1 / 4)][x144_0][(x126_1 % 4)] F.* x129[(x126_1 / 4)][x144_0][x126_0])))))))))
let x145 = (imap2 16 16 (\x146_0 x146_1 -> (let x147 = (imap3 4 16 16 (\x162_0 x162_1 x162_2 -> (isum1 4 (\x163_0 -> (x14[x162_0][x162_1][x163_0] F.* x17[x162_0][x162_2][x163_0])))))
in (let x148 = (imap3 4 16 16 (\x164_0 x164_1 x164_2 -> ((x147[x164_0][x164_1][x164_2] F./ fromi64 2) F.+ mask[x164_1][x164_2])))
in (let x149 = (imap3 4 16 4 (\x165_0 x165_1 x165_2 -> x122[x165_0][x165_1][x165_2]))
in (let x150 = (imap3 4 16 16 (\x166_0 x166_1 x166_2 -> (isum1 4 (\x167_0 -> (x149[x166_0][x166_1][x167_0] F.* x20[x166_0][x166_2][x167_0])))))
in (let x151 = (imap2 4 16 (\x168_0 x168_1 -> (imaximum1 16 (\x169_0 -> x148[x168_0][x168_1][x169_0]))))
in (let x152 = (imap3 4 16 16 (\x170_0 x170_1 x170_2 -> (F.exp (x148[x170_0][x170_1][x170_2] F.+ (F.neg x151[x170_0][x170_1])))))
in (let x153 = (imap2 4 16 (\x171_0 x171_1 -> (isum1 16 (\x172_0 -> x152[x171_0][x171_1][x172_0]))))
in (let x154 = (imap3 4 16 16 (\x173_0 x173_1 x173_2 -> x150[x173_0][x173_1][x173_2]))
in (let x155 = (imap2 4 16 (\x174_0 x174_1 -> (isum1 16 (\x175_0 -> (F.neg ((x154[x174_0][x174_1][x175_0] F.* x152[x174_0][x174_1][x175_0]) F.* (one F./ (x153[x174_0][x174_1] F.* x153[x174_0][x174_1]))))))))
in (let x156 = (imap3 4 16 16 (\x176_0 x176_1 x176_2 -> ((x154[x176_0][x176_1][x176_2] F.* (one F./ x153[x176_0][x176_1])) F.+ x155[x176_0][x176_1])))
in (let x157 = (imap2 4 16 (\x177_0 x177_1 -> (isum1 16 (\x178_0 -> (F.neg ((F.exp (x148[x177_0][x177_1][x178_0] F.+ (F.neg x151[x177_0][x177_1]))) F.* x156[x177_0][x177_1][x178_0]))))))
in (let x158 = (imap2 4 16 (\x179_0 x179_1 -> (imaximum1 16 (\x180_0 -> x148[x179_0][x179_1][x180_0]))))
in (let x159 = (imap2 4 16 (\x181_0 x181_1 -> (one F./ (isum1 16 (\x182_0 -> (one F.+ (F.neg (indicatorp (F.neg (x148[x181_0][x181_1][x182_0] F.+ (F.neg x158[x181_0][x181_1])))))))))))
in (let x160 = (imap3 4 16 16 (\x183_0 x183_1 x183_2 -> (((F.exp (x148[x183_0][x183_1][x183_2] F.+ (F.neg x151[x183_0][x183_1]))) F.* x156[x183_0][x183_1][x183_2]) F.+ ((x157[x183_0][x183_1] F.* (one F.+ (F.neg (indicatorp (F.neg (x148[x183_0][x183_1][x183_2] F.+ (F.neg x158[x183_0][x183_1]))))))) F.* x159[x183_0][x183_1]))))
in (let x161 = (imap3 4 16 16 (\x184_0 x184_1 x184_2 -> (x160[x184_0][x184_1][x184_2] F./ fromi64 2)))
in (isum1 16 (\x185_0 -> (x161[(x146_1 / 4)][x185_0][x146_0] F.* x14[(x146_1 / 4)][x185_0][(x146_1 % 4)]))))))))))))))))))))
let x186 = (imap2 16 16 (\x187_0 x187_1 -> (let x188 = (imap3 4 16 16 (\x203_0 x203_1 x203_2 -> (isum1 4 (\x204_0 -> (x14[x203_0][x203_1][x204_0] F.* x17[x203_0][x203_2][x204_0])))))
in (let x189 = (imap3 4 16 16 (\x205_0 x205_1 x205_2 -> ((x188[x205_0][x205_1][x205_2] F./ fromi64 2) F.+ mask[x205_1][x205_2])))
in (let x190 = (imap3 4 16 4 (\x206_0 x206_1 x206_2 -> x122[x206_0][x206_1][x206_2]))
in (let x191 = (imap3 4 16 16 (\x207_0 x207_1 x207_2 -> (isum1 4 (\x208_0 -> (x190[x207_0][x207_1][x208_0] F.* x20[x207_0][x207_2][x208_0])))))
in (let x192 = (imap2 4 16 (\x209_0 x209_1 -> (imaximum1 16 (\x210_0 -> x189[x209_0][x209_1][x210_0]))))
in (let x193 = (imap3 4 16 16 (\x211_0 x211_1 x211_2 -> (F.exp (x189[x211_0][x211_1][x211_2] F.+ (F.neg x192[x211_0][x211_1])))))
in (let x194 = (imap2 4 16 (\x212_0 x212_1 -> (isum1 16 (\x213_0 -> x193[x212_0][x212_1][x213_0]))))
in (let x195 = (imap3 4 16 16 (\x214_0 x214_1 x214_2 -> x191[x214_0][x214_1][x214_2]))
in (let x196 = (imap2 4 16 (\x215_0 x215_1 -> (isum1 16 (\x216_0 -> (F.neg ((x195[x215_0][x215_1][x216_0] F.* x193[x215_0][x215_1][x216_0]) F.* (one F./ (x194[x215_0][x215_1] F.* x194[x215_0][x215_1]))))))))
in (let x197 = (imap3 4 16 16 (\x217_0 x217_1 x217_2 -> ((x195[x217_0][x217_1][x217_2] F.* (one F./ x194[x217_0][x217_1])) F.+ x196[x217_0][x217_1])))
in (let x198 = (imap2 4 16 (\x218_0 x218_1 -> (isum1 16 (\x219_0 -> (F.neg ((F.exp (x189[x218_0][x218_1][x219_0] F.+ (F.neg x192[x218_0][x218_1]))) F.* x197[x218_0][x218_1][x219_0]))))))
in (let x199 = (imap2 4 16 (\x220_0 x220_1 -> (imaximum1 16 (\x221_0 -> x189[x220_0][x220_1][x221_0]))))
in (let x200 = (imap2 4 16 (\x222_0 x222_1 -> (one F./ (isum1 16 (\x223_0 -> (one F.+ (F.neg (indicatorp (F.neg (x189[x222_0][x222_1][x223_0] F.+ (F.neg x199[x222_0][x222_1])))))))))))
in (let x201 = (imap3 4 16 16 (\x224_0 x224_1 x224_2 -> (((F.exp (x189[x224_0][x224_1][x224_2] F.+ (F.neg x192[x224_0][x224_1]))) F.* x197[x224_0][x224_1][x224_2]) F.+ ((x198[x224_0][x224_1] F.* (one F.+ (F.neg (indicatorp (F.neg (x189[x224_0][x224_1][x224_2] F.+ (F.neg x199[x224_0][x224_1]))))))) F.* x200[x224_0][x224_1]))))
in (let x202 = (imap3 4 16 16 (\x225_0 x225_1 x225_2 -> (x201[x225_0][x225_1][x225_2] F./ fromi64 2)))
in (isum1 16 (\x226_0 -> (x202[(x187_1 / 4)][x187_0][x226_0] F.* x17[(x187_1 / 4)][x226_0][(x187_1 % 4)]))))))))))))))))))))
let x227 = (let x228 = (imap1 16 (\x234_0 -> ((isum1 16 (\x235_0 -> (x0[x234_0][x235_0] F.* x0[x234_0][x235_0]))) F./ fromi64 16)))
in (let x229 = (imap1 16 (\x236_0 -> (F.sqrt (x228[x236_0] F.+ (one F./ fromi64 100000)))))
in (let x230 = (imap2 16 16 (\x237_0 x237_1 -> (((isum1 16 (\x238_0 -> (x125[x237_0][x238_0] F.* wval[x238_0][x237_1]))) F.+ (isum1 16 (\x239_0 -> (x145[x237_0][x239_0] F.* wkey[x239_0][x237_1])))) F.+ (isum1 16 (\x240_0 -> (x186[x237_0][x240_0] F.* wqry[x240_0][x237_1]))))))
in (let x231 = (imap1 16 (\x241_0 -> (isum1 16 (\x242_0 -> (F.neg ((x230[x241_0][x242_0] F.* x0[x241_0][x242_0]) F.* (one F./ (x229[x241_0] F.* x229[x241_0]))))))))
in (let x232 = (imap1 16 (\x243_0 -> (x231[x243_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x228[x243_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x233_0 x233_1 -> (x107[x233_0][x233_1] F.+ ((x230[x233_0][x233_1] F.* (one F./ x229[x233_0])) F.+ (((x232[x233_0] F./ fromi64 16) F.* x0[x233_0][x233_1]) F.+ ((x232[x233_0] F./ fromi64 16) F.* x0[x233_0][x233_1])))))))))))

let dmask = (let x244 = (imap3 4 16 16 (\x260_0 x260_1 x260_2 -> (isum1 4 (\x261_0 -> (x14[x260_0][x260_1][x261_0] F.* x17[x260_0][x260_2][x261_0])))))
in (let x245 = (imap3 4 16 16 (\x262_0 x262_1 x262_2 -> ((x244[x262_0][x262_1][x262_2] F./ fromi64 2) F.+ mask[x262_1][x262_2])))
in (let x246 = (imap3 4 16 4 (\x263_0 x263_1 x263_2 -> x122[x263_0][x263_1][x263_2]))
in (let x247 = (imap3 4 16 16 (\x264_0 x264_1 x264_2 -> (isum1 4 (\x265_0 -> (x246[x264_0][x264_1][x265_0] F.* x20[x264_0][x264_2][x265_0])))))
in (let x248 = (imap2 4 16 (\x266_0 x266_1 -> (imaximum1 16 (\x267_0 -> x245[x266_0][x266_1][x267_0]))))
in (let x249 = (imap3 4 16 16 (\x268_0 x268_1 x268_2 -> (F.exp (x245[x268_0][x268_1][x268_2] F.+ (F.neg x248[x268_0][x268_1])))))
in (let x250 = (imap2 4 16 (\x269_0 x269_1 -> (isum1 16 (\x270_0 -> x249[x269_0][x269_1][x270_0]))))
in (let x251 = (imap3 4 16 16 (\x271_0 x271_1 x271_2 -> x247[x271_0][x271_1][x271_2]))
in (let x252 = (imap2 4 16 (\x272_0 x272_1 -> (isum1 16 (\x273_0 -> (F.neg ((x251[x272_0][x272_1][x273_0] F.* x249[x272_0][x272_1][x273_0]) F.* (one F./ (x250[x272_0][x272_1] F.* x250[x272_0][x272_1]))))))))
in (let x253 = (imap3 4 16 16 (\x274_0 x274_1 x274_2 -> ((x251[x274_0][x274_1][x274_2] F.* (one F./ x250[x274_0][x274_1])) F.+ x252[x274_0][x274_1])))
in (let x254 = (imap2 4 16 (\x275_0 x275_1 -> (isum1 16 (\x276_0 -> (F.neg ((F.exp (x245[x275_0][x275_1][x276_0] F.+ (F.neg x248[x275_0][x275_1]))) F.* x253[x275_0][x275_1][x276_0]))))))
in (let x255 = (imap2 4 16 (\x277_0 x277_1 -> (imaximum1 16 (\x278_0 -> x245[x277_0][x277_1][x278_0]))))
in (let x256 = (imap2 4 16 (\x279_0 x279_1 -> (one F./ (isum1 16 (\x280_0 -> (one F.+ (F.neg (indicatorp (F.neg (x245[x279_0][x279_1][x280_0] F.+ (F.neg x255[x279_0][x279_1])))))))))))
in (let x257 = (imap3 4 16 16 (\x281_0 x281_1 x281_2 -> (((F.exp (x245[x281_0][x281_1][x281_2] F.+ (F.neg x248[x281_0][x281_1]))) F.* x253[x281_0][x281_1][x281_2]) F.+ ((x254[x281_0][x281_1] F.* (one F.+ (F.neg (indicatorp (F.neg (x245[x281_0][x281_1][x281_2] F.+ (F.neg x255[x281_0][x281_1]))))))) F.* x256[x281_0][x281_1]))))
in (imap2 16 16 (\x259_0 x259_1 -> (isum1 4 (\x258_0 -> x257[x258_0][x259_0][x259_1]))))))))))))))))))
let dwpe = (let x282 = (imap1 16 (\x288_0 -> ((isum1 16 (\x289_0 -> ((wpe[x288_0][x289_0] F.+ wseq[x288_0][x289_0]) F.* (wpe[x288_0][x289_0] F.+ wseq[x288_0][x289_0])))) F./ fromi64 16)))
in (let x283 = (imap1 16 (\x290_0 -> (F.sqrt (x282[x290_0] F.+ (one F./ fromi64 100000)))))
in (let x284 = (imap2 16 16 (\x291_0 x291_1 -> x227[x291_0][x291_1]))
in (let x285 = (imap1 16 (\x292_0 -> (isum1 16 (\x293_0 -> (F.neg ((x284[x292_0][x293_0] F.* (wpe[x292_0][x293_0] F.+ wseq[x292_0][x293_0])) F.* (one F./ (x283[x292_0] F.* x283[x292_0]))))))))
in (let x286 = (imap1 16 (\x294_0 -> (x285[x294_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x282[x294_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x287_0 x287_1 -> ((x284[x287_0][x287_1] F.* (one F./ x283[x287_0])) F.+ (((x286[x287_0] F./ fromi64 16) F.* (wpe[x287_0][x287_1] F.+ wseq[x287_0][x287_1])) F.+ ((x286[x287_0] F./ fromi64 16) F.* (wpe[x287_0][x287_1] F.+ wseq[x287_0][x287_1])))))))))))
let dwqry = (imap2 16 16 (\x295_0 x295_1 -> (isum1 16 (\x296_0 -> (x186[x296_0][x295_0] F.* x7[x296_0][x295_1])))))
let dwkey = (imap2 16 16 (\x297_0 x297_1 -> (isum1 16 (\x298_0 -> (x145[x298_0][x297_0] F.* x7[x298_0][x297_1])))))
let dwval = (imap2 16 16 (\x299_0 x299_1 -> (isum1 16 (\x300_0 -> (x125[x300_0][x299_0] F.* x7[x300_0][x299_1])))))
let dwout = (imap2 16 16 (\x301_0 x301_1 -> (isum1 16 (\x302_0 -> (x107[x302_0][x301_0] F.* x23[x302_0][x301_1])))))
let dwup = (imap2 64 16 (\x303_0 x303_1 -> (isum1 16 (\x304_0 -> (x104[x304_0][x303_0] F.* x46[x304_0][x303_1])))))
let dwdown = (imap2 16 64 (\x305_0 x305_1 -> (isum1 16 (\x306_0 -> (x101[x306_0][x305_0] F.* x56[x306_0][x305_1])))))
let dwvoc = (imap2 27 16 (\x307_0 x307_1 -> (isum1 16 (\x308_0 -> (x66[x308_0][x307_0] F.* x58[x308_0][x307_1])))))
let dwseq = (let x309 = (imap1 16 (\x315_0 -> ((isum1 16 (\x316_0 -> ((wpe[x315_0][x316_0] F.+ wseq[x315_0][x316_0]) F.* (wpe[x315_0][x316_0] F.+ wseq[x315_0][x316_0])))) F./ fromi64 16)))
in (let x310 = (imap1 16 (\x317_0 -> (F.sqrt (x309[x317_0] F.+ (one F./ fromi64 100000)))))
in (let x311 = (imap2 16 16 (\x318_0 x318_1 -> x227[x318_0][x318_1]))
in (let x312 = (imap1 16 (\x319_0 -> (isum1 16 (\x320_0 -> (F.neg ((x311[x319_0][x320_0] F.* (wpe[x319_0][x320_0] F.+ wseq[x319_0][x320_0])) F.* (one F./ (x310[x319_0] F.* x310[x319_0]))))))))
in (let x313 = (imap1 16 (\x321_0 -> (x312[x321_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x309[x321_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x314_0 x314_1 -> ((x311[x314_0][x314_1] F.* (one F./ x310[x314_0])) F.+ (((x313[x314_0] F./ fromi64 16) F.* (wpe[x314_0][x314_1] F.+ wseq[x314_0][x314_1])) F.+ ((x313[x314_0] F./ fromi64 16) F.* (wpe[x314_0][x314_1] F.+ wseq[x314_0][x314_1])))))))))))
let dtarget = (imap2 16 27 (\x322_0 x322_1 -> (let x323 = (imaximum1 27 (\x327_0 -> x61[x322_0][x327_0]))
in (let x324 = (imap1 27 (\x328_0 -> (F.exp (x61[x322_0][x328_0] F.+ (F.neg x323)))))
in (let x325 = (isum1 27 (\x329_0 -> x324[x329_0]))
in (let x326 = (imap1 27 (\x330_0 -> (x324[x330_0] F.* (one F./ x325))))
in ((F.neg x64[x322_0]) F.* (F.log x326[x322_1]))))))))

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