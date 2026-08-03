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

let x0 = (let x1 = (imap1 16 (\x5_0 -> ((isum1 16 (\x6_0 -> ((wpe[x5_0][x6_0] F.+ wseq[x5_0][x6_0]) F.* (wpe[x5_0][x6_0] F.+ wseq[x5_0][x6_0])))) F./ fromi64 16)))
in (let x2 = (imap1 16 (\x7_0 -> (F.sqrt (x1[x7_0] F.+ (one F./ fromi64 100000)))))
in (let x3 = (imap2 16 16 (\x8_0 x8_1 -> ((wpe[x8_0][x8_1] F.+ wseq[x8_0][x8_1]) F.* (one F./ x2[x8_0]))))
in (imap2 16 16 (\x4_0 x4_1 -> x3[x4_0][x4_1])))))
let x9 = (let x10 = (imap1 16 (\x14_0 -> ((isum1 16 (\x15_0 -> (x0[x14_0][x15_0] F.* x0[x14_0][x15_0]))) F./ fromi64 16)))
in (let x11 = (imap1 16 (\x16_0 -> (F.sqrt (x10[x16_0] F.+ (one F./ fromi64 100000)))))
in (let x12 = (imap2 16 16 (\x17_0 x17_1 -> (x0[x17_0][x17_1] F.* (one F./ x11[x17_0]))))
in (imap2 16 16 (\x13_0 x13_1 -> x12[x13_0][x13_1])))))
let x18 = (imap3 4 16 4 (\x19_0 x19_1 x19_2 -> (isum1 16 (\x20_0 -> (wqry[((x19_0 * 4) + x19_2)][x20_0] F.* x9[x19_1][x20_0])))))
let x21 = (imap3 4 16 4 (\x22_0 x22_1 x22_2 -> (isum1 16 (\x23_0 -> (wkey[((x22_0 * 4) + x22_2)][x23_0] F.* x9[x22_1][x23_0])))))
let x24 = (imap3 4 16 4 (\x25_0 x25_1 x25_2 -> (isum1 16 (\x26_0 -> (wval[((x25_0 * 4) + x25_2)][x26_0] F.* x9[x25_1][x26_0])))))
let x27 = (let x28 = (imap7 16 4 16 4 4 16 16 (\x37_0 x37_1 x37_2 x37_3 x37_4 x37_5 x37_6 -> (isum1 4 (\x38_0 -> (x18[x37_4][x37_5][x38_0] F.* x21[x37_4][x37_6][x38_0])))))
in (let x29 = (imap7 16 4 16 4 4 16 16 (\x39_0 x39_1 x39_2 x39_3 x39_4 x39_5 x39_6 -> ((x28[x39_0][x39_1][x39_2][x39_3][x39_4][x39_5][x39_6] F./ fromi64 2) F.+ mask[x39_5][x39_6])))
in (let x30 = (imap6 16 4 16 4 4 16 (\x40_0 x40_1 x40_2 x40_3 x40_4 x40_5 -> (imaximum1 16 (\x41_0 -> x29[x40_0][x40_1][x40_2][x40_3][x40_4][x40_5][x41_0]))))
in (let x31 = (imap7 16 4 16 4 4 16 16 (\x42_0 x42_1 x42_2 x42_3 x42_4 x42_5 x42_6 -> (F.exp (x29[x42_0][x42_1][x42_2][x42_3][x42_4][x42_5][x42_6] F.+ (F.neg x30[x42_0][x42_1][x42_2][x42_3][x42_4][x42_5])))))
in (let x32 = (imap6 16 4 16 4 4 16 (\x43_0 x43_1 x43_2 x43_3 x43_4 x43_5 -> (isum1 16 (\x44_0 -> x31[x43_0][x43_1][x43_2][x43_3][x43_4][x43_5][x44_0]))))
in (let x33 = (imap7 16 4 16 4 4 16 16 (\x45_0 x45_1 x45_2 x45_3 x45_4 x45_5 x45_6 -> (x31[x45_0][x45_1][x45_2][x45_3][x45_4][x45_5][x45_6] F.* (one F./ x32[x45_0][x45_1][x45_2][x45_3][x45_4][x45_5]))))
in (let x34 = (imap7 16 4 16 4 4 16 16 (\x46_0 x46_1 x46_2 x46_3 x46_4 x46_5 x46_6 -> x33[x46_0][x46_1][x46_2][x46_3][x46_4][x46_5][x46_6]))
in (let x35 = (imap7 16 4 16 4 4 16 4 (\x47_0 x47_1 x47_2 x47_3 x47_4 x47_5 x47_6 -> (isum1 16 (\x48_0 -> (x34[x47_0][x47_1][x47_2][x47_3][x47_4][x47_5][x48_0] F.* x24[x47_4][x48_0][x47_6])))))
in (imap2 16 16 (\x36_0 x36_1 -> x35[x36_0][(x36_1 / 4)][x36_0][(x36_1 / 4)][(x36_1 / 4)][x36_0][(x36_1 % 4)]))))))))))
let x49 = (imap2 16 16 (\x50_0 x50_1 -> ((isum1 16 (\x51_0 -> (wout[x50_1][x51_0] F.* x27[x50_0][x51_0]))) F.+ x0[x50_0][x50_1])))
let x52 = (let x53 = (imap1 16 (\x57_0 -> ((isum1 16 (\x58_0 -> (x49[x57_0][x58_0] F.* x49[x57_0][x58_0]))) F./ fromi64 16)))
in (let x54 = (imap1 16 (\x59_0 -> (F.sqrt (x53[x59_0] F.+ (one F./ fromi64 100000)))))
in (let x55 = (imap2 16 16 (\x60_0 x60_1 -> (x49[x60_0][x60_1] F.* (one F./ x54[x60_0]))))
in (imap2 16 16 (\x56_0 x56_1 -> x55[x56_0][x56_1])))))
let x61 = (imap2 16 64 (\x62_0 x62_1 -> (isum1 16 (\x63_0 -> (wup[x62_1][x63_0] F.* x52[x62_0][x63_0])))))
let x64 = (imap2 16 64 (\x65_0 x65_1 -> F.max x61[x65_0][x65_1] zero))
let x66 = (imap2 16 16 (\x67_0 x67_1 -> ((isum1 64 (\x68_0 -> (wdown[x67_1][x68_0] F.* x64[x67_0][x68_0]))) F.+ x49[x67_0][x67_1])))
let x69 = (imap2 16 27 (\x70_0 x70_1 -> (isum1 16 (\x71_0 -> (wvoc[x70_1][x71_0] F.* x66[x70_0][x71_0])))))
let x72 = (imap1 16 (\x73_0 -> (one F./ fromi64 16)))
let x74 = (let x75 = (imap2 16 27 (\x85_0 x85_1 -> (imaximum1 27 (\x86_0 -> x69[x85_0][x86_0]))))
in (let x76 = (imap3 16 27 27 (\x87_0 x87_1 x87_2 -> (F.exp (x69[x87_0][x87_2] F.+ (F.neg x75[x87_0][x87_1])))))
in (let x77 = (imap2 16 27 (\x88_0 x88_1 -> (isum1 27 (\x89_0 -> x76[x88_0][x88_1][x89_0]))))
in (let x78 = (imap3 16 27 27 (\x90_0 x90_1 x90_2 -> (if ((x90_2 == x90_1)) then (let x91 = (imaximum1 27 (\x95_0 -> x69[x90_0][x95_0]))
in (let x92 = (imap1 27 (\x96_0 -> (F.exp (x69[x90_0][x96_0] F.+ (F.neg x91)))))
in (let x93 = (isum1 27 (\x97_0 -> x92[x97_0]))
in (let x94 = (imap1 27 (\x98_0 -> (x92[x98_0] F.* (one F./ x93))))
in (((F.neg x72[x90_0]) F.* target[x90_0][x90_1]) F.* (one F./ x94[x90_2])))))) else zero)))
in (let x79 = (imap2 16 27 (\x99_0 x99_1 -> (isum1 27 (\x100_0 -> (F.neg ((x78[x99_0][x99_1][x100_0] F.* x76[x99_0][x99_1][x100_0]) F.* (one F./ (x77[x99_0][x99_1] F.* x77[x99_0][x99_1]))))))))
in (let x80 = (imap3 16 27 27 (\x101_0 x101_1 x101_2 -> ((x78[x101_0][x101_1][x101_2] F.* (one F./ x77[x101_0][x101_1])) F.+ x79[x101_0][x101_1])))
in (let x81 = (imap2 16 27 (\x102_0 x102_1 -> (isum1 27 (\x103_0 -> (F.neg ((F.exp (x69[x102_0][x103_0] F.+ (F.neg x75[x102_0][x102_1]))) F.* x80[x102_0][x102_1][x103_0]))))))
in (let x82 = (imap2 16 27 (\x104_0 x104_1 -> (imaximum1 27 (\x105_0 -> x69[x104_0][x105_0]))))
in (let x83 = (imap2 16 27 (\x106_0 x106_1 -> (one F./ ((isum1 27 (\x107_0 -> one)) F.+ (isum1 27 (\x108_0 -> (F.neg (indicatorp (F.neg (x69[x106_0][x108_0] F.+ (F.neg x82[x106_0][x106_1])))))))))))
in (imap2 16 27 (\x84_0 x84_1 -> ((isum1 27 (\x109_0 -> ((F.exp (x69[x84_0][x84_1] F.+ (F.neg x75[x84_0][x109_0]))) F.* x80[x84_0][x109_0][x84_1]))) F.+ (isum1 27 (\x110_0 -> ((x81[x84_0][x110_0] F.* (one F.+ (F.neg (indicatorp (F.neg (x69[x84_0][x84_1] F.+ (F.neg x82[x84_0][x110_0]))))))) F.* x83[x84_0][x110_0])))))))))))))))
let x111 = (imap2 16 16 (\x112_0 x112_1 -> (isum1 27 (\x113_0 -> (x74[x112_0][x113_0] F.* wvoc[x113_0][x112_1])))))
let x114 = (imap2 16 64 (\x115_0 x115_1 -> ((indicatorp x61[x115_0][x115_1]) F.* (isum1 16 (\x116_0 -> (x111[x115_0][x116_0] F.* wdown[x116_0][x115_1]))))))
let x117 = (let x118 = (imap1 16 (\x124_0 -> ((isum1 16 (\x125_0 -> (x49[x124_0][x125_0] F.* x49[x124_0][x125_0]))) F./ fromi64 16)))
in (let x119 = (imap1 16 (\x126_0 -> (F.sqrt (x118[x126_0] F.+ (one F./ fromi64 100000)))))
in (let x120 = (imap2 16 16 (\x127_0 x127_1 -> (isum1 64 (\x128_0 -> (x114[x127_0][x128_0] F.* wup[x128_0][x127_1])))))
in (let x121 = (imap1 16 (\x129_0 -> (isum1 16 (\x130_0 -> (F.neg ((x120[x129_0][x130_0] F.* x49[x129_0][x130_0]) F.* (one F./ (x119[x129_0] F.* x119[x129_0]))))))))
in (let x122 = (imap1 16 (\x131_0 -> (x121[x131_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x118[x131_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x123_0 x123_1 -> (x111[x123_0][x123_1] F.+ ((x120[x123_0][x123_1] F.* (one F./ x119[x123_0])) F.+ (((x122[x123_0] F./ fromi64 16) F.* x49[x123_0][x123_1]) F.+ ((x122[x123_0] F./ fromi64 16) F.* x49[x123_0][x123_1])))))))))))
let x132 = (imap3 4 16 4 (\x133_0 x133_1 x133_2 -> (isum1 16 (\x134_0 -> (x117[x133_1][x134_0] F.* wout[x134_0][((x133_0 * 4) + x133_2)])))))
let x135 = (let x136 = (imap5 16 4 4 16 16 (\x145_0 x145_1 x145_2 x145_3 x145_4 -> (isum1 4 (\x146_0 -> (x18[x145_2][x145_3][x146_0] F.* x21[x145_2][x145_4][x146_0])))))
in (let x137 = (imap5 16 4 4 16 16 (\x147_0 x147_1 x147_2 x147_3 x147_4 -> ((x136[x147_0][x147_1][x147_2][x147_3][x147_4] F./ fromi64 2) F.+ mask[x147_3][x147_4])))
in (let x138 = (imap4 16 4 4 16 (\x148_0 x148_1 x148_2 x148_3 -> (imaximum1 16 (\x149_0 -> x137[x148_0][x148_1][x148_2][x148_3][x149_0]))))
in (let x139 = (imap5 16 4 4 16 16 (\x150_0 x150_1 x150_2 x150_3 x150_4 -> (F.exp (x137[x150_0][x150_1][x150_2][x150_3][x150_4] F.+ (F.neg x138[x150_0][x150_1][x150_2][x150_3])))))
in (let x140 = (imap4 16 4 4 16 (\x151_0 x151_1 x151_2 x151_3 -> (isum1 16 (\x152_0 -> x139[x151_0][x151_1][x151_2][x151_3][x152_0]))))
in (let x141 = (imap5 16 4 4 16 16 (\x153_0 x153_1 x153_2 x153_3 x153_4 -> (x139[x153_0][x153_1][x153_2][x153_3][x153_4] F.* (one F./ x140[x153_0][x153_1][x153_2][x153_3]))))
in (let x142 = (imap5 16 4 4 16 16 (\x154_0 x154_1 x154_2 x154_3 x154_4 -> x141[x154_0][x154_1][x154_2][x154_3][x154_4]))
in (let x143 = (imap5 16 4 4 16 4 (\x155_0 x155_1 x155_2 x155_3 x155_4 -> x132[x155_2][x155_3][x155_4]))
in (imap2 16 16 (\x144_0 x144_1 -> (isum1 16 (\x156_0 -> (x143[x144_0][(x144_1 / 4)][(x144_1 / 4)][x156_0][(x144_1 % 4)] F.* x142[x144_0][(x144_1 / 4)][(x144_1 / 4)][x156_0][x144_0])))))))))))))
let x157 = (let x158 = (imap5 16 4 4 16 16 (\x174_0 x174_1 x174_2 x174_3 x174_4 -> (isum1 4 (\x175_0 -> (x18[x174_2][x174_3][x175_0] F.* x21[x174_2][x174_4][x175_0])))))
in (let x159 = (imap5 16 4 4 16 16 (\x176_0 x176_1 x176_2 x176_3 x176_4 -> ((x158[x176_0][x176_1][x176_2][x176_3][x176_4] F./ fromi64 2) F.+ mask[x176_3][x176_4])))
in (let x160 = (imap5 16 4 4 16 4 (\x177_0 x177_1 x177_2 x177_3 x177_4 -> x132[x177_2][x177_3][x177_4]))
in (let x161 = (imap5 16 4 4 16 16 (\x178_0 x178_1 x178_2 x178_3 x178_4 -> (isum1 4 (\x179_0 -> (x160[x178_0][x178_1][x178_2][x178_3][x179_0] F.* x24[x178_2][x178_4][x179_0])))))
in (let x162 = (imap4 16 4 4 16 (\x180_0 x180_1 x180_2 x180_3 -> (imaximum1 16 (\x181_0 -> x159[x180_0][x180_1][x180_2][x180_3][x181_0]))))
in (let x163 = (imap5 16 4 4 16 16 (\x182_0 x182_1 x182_2 x182_3 x182_4 -> (F.exp (x159[x182_0][x182_1][x182_2][x182_3][x182_4] F.+ (F.neg x162[x182_0][x182_1][x182_2][x182_3])))))
in (let x164 = (imap4 16 4 4 16 (\x183_0 x183_1 x183_2 x183_3 -> (isum1 16 (\x184_0 -> x163[x183_0][x183_1][x183_2][x183_3][x184_0]))))
in (let x165 = (imap5 16 4 4 16 16 (\x185_0 x185_1 x185_2 x185_3 x185_4 -> x161[x185_0][x185_1][x185_2][x185_3][x185_4]))
in (let x166 = (imap4 16 4 4 16 (\x186_0 x186_1 x186_2 x186_3 -> (isum1 16 (\x187_0 -> (F.neg ((x165[x186_0][x186_1][x186_2][x186_3][x187_0] F.* x163[x186_0][x186_1][x186_2][x186_3][x187_0]) F.* (one F./ (x164[x186_0][x186_1][x186_2][x186_3] F.* x164[x186_0][x186_1][x186_2][x186_3]))))))))
in (let x167 = (imap5 16 4 4 16 16 (\x188_0 x188_1 x188_2 x188_3 x188_4 -> ((x165[x188_0][x188_1][x188_2][x188_3][x188_4] F.* (one F./ x164[x188_0][x188_1][x188_2][x188_3])) F.+ x166[x188_0][x188_1][x188_2][x188_3])))
in (let x168 = (imap4 16 4 4 16 (\x189_0 x189_1 x189_2 x189_3 -> (isum1 16 (\x190_0 -> (F.neg ((F.exp (x159[x189_0][x189_1][x189_2][x189_3][x190_0] F.+ (F.neg x162[x189_0][x189_1][x189_2][x189_3]))) F.* x167[x189_0][x189_1][x189_2][x189_3][x190_0]))))))
in (let x169 = (imap4 16 4 4 16 (\x191_0 x191_1 x191_2 x191_3 -> (imaximum1 16 (\x192_0 -> x159[x191_0][x191_1][x191_2][x191_3][x192_0]))))
in (let x170 = (imap4 16 4 4 16 (\x193_0 x193_1 x193_2 x193_3 -> (one F./ ((isum1 16 (\x194_0 -> one)) F.+ (isum1 16 (\x195_0 -> (F.neg (indicatorp (F.neg (x159[x193_0][x193_1][x193_2][x193_3][x195_0] F.+ (F.neg x169[x193_0][x193_1][x193_2][x193_3])))))))))))
in (let x171 = (imap5 16 4 4 16 16 (\x196_0 x196_1 x196_2 x196_3 x196_4 -> (((F.exp (x159[x196_0][x196_1][x196_2][x196_3][x196_4] F.+ (F.neg x162[x196_0][x196_1][x196_2][x196_3]))) F.* x167[x196_0][x196_1][x196_2][x196_3][x196_4]) F.+ ((x168[x196_0][x196_1][x196_2][x196_3] F.* (one F.+ (F.neg (indicatorp (F.neg (x159[x196_0][x196_1][x196_2][x196_3][x196_4] F.+ (F.neg x169[x196_0][x196_1][x196_2][x196_3]))))))) F.* x170[x196_0][x196_1][x196_2][x196_3]))))
in (let x172 = (imap5 16 4 4 16 16 (\x197_0 x197_1 x197_2 x197_3 x197_4 -> (x171[x197_0][x197_1][x197_2][x197_3][x197_4] F./ fromi64 2)))
in (imap2 16 16 (\x173_0 x173_1 -> (isum1 16 (\x198_0 -> (x172[x173_0][(x173_1 / 4)][(x173_1 / 4)][x198_0][x173_0] F.* x18[(x173_1 / 4)][x198_0][(x173_1 % 4)]))))))))))))))))))))
let x199 = (let x200 = (imap5 16 4 4 16 16 (\x216_0 x216_1 x216_2 x216_3 x216_4 -> (isum1 4 (\x217_0 -> (x18[x216_2][x216_3][x217_0] F.* x21[x216_2][x216_4][x217_0])))))
in (let x201 = (imap5 16 4 4 16 16 (\x218_0 x218_1 x218_2 x218_3 x218_4 -> ((x200[x218_0][x218_1][x218_2][x218_3][x218_4] F./ fromi64 2) F.+ mask[x218_3][x218_4])))
in (let x202 = (imap5 16 4 4 16 4 (\x219_0 x219_1 x219_2 x219_3 x219_4 -> x132[x219_2][x219_3][x219_4]))
in (let x203 = (imap5 16 4 4 16 16 (\x220_0 x220_1 x220_2 x220_3 x220_4 -> (isum1 4 (\x221_0 -> (x202[x220_0][x220_1][x220_2][x220_3][x221_0] F.* x24[x220_2][x220_4][x221_0])))))
in (let x204 = (imap4 16 4 4 16 (\x222_0 x222_1 x222_2 x222_3 -> (imaximum1 16 (\x223_0 -> x201[x222_0][x222_1][x222_2][x222_3][x223_0]))))
in (let x205 = (imap5 16 4 4 16 16 (\x224_0 x224_1 x224_2 x224_3 x224_4 -> (F.exp (x201[x224_0][x224_1][x224_2][x224_3][x224_4] F.+ (F.neg x204[x224_0][x224_1][x224_2][x224_3])))))
in (let x206 = (imap4 16 4 4 16 (\x225_0 x225_1 x225_2 x225_3 -> (isum1 16 (\x226_0 -> x205[x225_0][x225_1][x225_2][x225_3][x226_0]))))
in (let x207 = (imap5 16 4 4 16 16 (\x227_0 x227_1 x227_2 x227_3 x227_4 -> x203[x227_0][x227_1][x227_2][x227_3][x227_4]))
in (let x208 = (imap4 16 4 4 16 (\x228_0 x228_1 x228_2 x228_3 -> (isum1 16 (\x229_0 -> (F.neg ((x207[x228_0][x228_1][x228_2][x228_3][x229_0] F.* x205[x228_0][x228_1][x228_2][x228_3][x229_0]) F.* (one F./ (x206[x228_0][x228_1][x228_2][x228_3] F.* x206[x228_0][x228_1][x228_2][x228_3]))))))))
in (let x209 = (imap5 16 4 4 16 16 (\x230_0 x230_1 x230_2 x230_3 x230_4 -> ((x207[x230_0][x230_1][x230_2][x230_3][x230_4] F.* (one F./ x206[x230_0][x230_1][x230_2][x230_3])) F.+ x208[x230_0][x230_1][x230_2][x230_3])))
in (let x210 = (imap4 16 4 4 16 (\x231_0 x231_1 x231_2 x231_3 -> (isum1 16 (\x232_0 -> (F.neg ((F.exp (x201[x231_0][x231_1][x231_2][x231_3][x232_0] F.+ (F.neg x204[x231_0][x231_1][x231_2][x231_3]))) F.* x209[x231_0][x231_1][x231_2][x231_3][x232_0]))))))
in (let x211 = (imap4 16 4 4 16 (\x233_0 x233_1 x233_2 x233_3 -> (imaximum1 16 (\x234_0 -> x201[x233_0][x233_1][x233_2][x233_3][x234_0]))))
in (let x212 = (imap4 16 4 4 16 (\x235_0 x235_1 x235_2 x235_3 -> (one F./ ((isum1 16 (\x236_0 -> one)) F.+ (isum1 16 (\x237_0 -> (F.neg (indicatorp (F.neg (x201[x235_0][x235_1][x235_2][x235_3][x237_0] F.+ (F.neg x211[x235_0][x235_1][x235_2][x235_3])))))))))))
in (let x213 = (imap5 16 4 4 16 16 (\x238_0 x238_1 x238_2 x238_3 x238_4 -> (((F.exp (x201[x238_0][x238_1][x238_2][x238_3][x238_4] F.+ (F.neg x204[x238_0][x238_1][x238_2][x238_3]))) F.* x209[x238_0][x238_1][x238_2][x238_3][x238_4]) F.+ ((x210[x238_0][x238_1][x238_2][x238_3] F.* (one F.+ (F.neg (indicatorp (F.neg (x201[x238_0][x238_1][x238_2][x238_3][x238_4] F.+ (F.neg x211[x238_0][x238_1][x238_2][x238_3]))))))) F.* x212[x238_0][x238_1][x238_2][x238_3]))))
in (let x214 = (imap5 16 4 4 16 16 (\x239_0 x239_1 x239_2 x239_3 x239_4 -> (x213[x239_0][x239_1][x239_2][x239_3][x239_4] F./ fromi64 2)))
in (imap2 16 16 (\x215_0 x215_1 -> (isum1 16 (\x240_0 -> (x214[x215_0][(x215_1 / 4)][(x215_1 / 4)][x215_0][x240_0] F.* x21[(x215_1 / 4)][x240_0][(x215_1 % 4)]))))))))))))))))))))
let x241 = (let x242 = (imap1 16 (\x248_0 -> ((isum1 16 (\x249_0 -> (x0[x248_0][x249_0] F.* x0[x248_0][x249_0]))) F./ fromi64 16)))
in (let x243 = (imap1 16 (\x250_0 -> (F.sqrt (x242[x250_0] F.+ (one F./ fromi64 100000)))))
in (let x244 = (imap2 16 16 (\x251_0 x251_1 -> (((isum1 16 (\x252_0 -> (x135[x251_0][x252_0] F.* wval[x252_0][x251_1]))) F.+ (isum1 16 (\x253_0 -> (x157[x251_0][x253_0] F.* wkey[x253_0][x251_1])))) F.+ (isum1 16 (\x254_0 -> (x199[x251_0][x254_0] F.* wqry[x254_0][x251_1]))))))
in (let x245 = (imap1 16 (\x255_0 -> (isum1 16 (\x256_0 -> (F.neg ((x244[x255_0][x256_0] F.* x0[x255_0][x256_0]) F.* (one F./ (x243[x255_0] F.* x243[x255_0]))))))))
in (let x246 = (imap1 16 (\x257_0 -> (x245[x257_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x242[x257_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x247_0 x247_1 -> (x117[x247_0][x247_1] F.+ ((x244[x247_0][x247_1] F.* (one F./ x243[x247_0])) F.+ (((x246[x247_0] F./ fromi64 16) F.* x0[x247_0][x247_1]) F.+ ((x246[x247_0] F./ fromi64 16) F.* x0[x247_0][x247_1])))))))))))

let dmask = (let x258 = (imap3 4 16 16 (\x274_0 x274_1 x274_2 -> (isum1 4 (\x275_0 -> (x18[x274_0][x274_1][x275_0] F.* x21[x274_0][x274_2][x275_0])))))
in (let x259 = (imap3 4 16 16 (\x276_0 x276_1 x276_2 -> ((x258[x276_0][x276_1][x276_2] F./ fromi64 2) F.+ mask[x276_1][x276_2])))
in (let x260 = (imap3 4 16 4 (\x277_0 x277_1 x277_2 -> x132[x277_0][x277_1][x277_2]))
in (let x261 = (imap3 4 16 16 (\x278_0 x278_1 x278_2 -> (isum1 4 (\x279_0 -> (x260[x278_0][x278_1][x279_0] F.* x24[x278_0][x278_2][x279_0])))))
in (let x262 = (imap2 4 16 (\x280_0 x280_1 -> (imaximum1 16 (\x281_0 -> x259[x280_0][x280_1][x281_0]))))
in (let x263 = (imap3 4 16 16 (\x282_0 x282_1 x282_2 -> (F.exp (x259[x282_0][x282_1][x282_2] F.+ (F.neg x262[x282_0][x282_1])))))
in (let x264 = (imap2 4 16 (\x283_0 x283_1 -> (isum1 16 (\x284_0 -> x263[x283_0][x283_1][x284_0]))))
in (let x265 = (imap3 4 16 16 (\x285_0 x285_1 x285_2 -> x261[x285_0][x285_1][x285_2]))
in (let x266 = (imap2 4 16 (\x286_0 x286_1 -> (isum1 16 (\x287_0 -> (F.neg ((x265[x286_0][x286_1][x287_0] F.* x263[x286_0][x286_1][x287_0]) F.* (one F./ (x264[x286_0][x286_1] F.* x264[x286_0][x286_1]))))))))
in (let x267 = (imap3 4 16 16 (\x288_0 x288_1 x288_2 -> ((x265[x288_0][x288_1][x288_2] F.* (one F./ x264[x288_0][x288_1])) F.+ x266[x288_0][x288_1])))
in (let x268 = (imap2 4 16 (\x289_0 x289_1 -> (isum1 16 (\x290_0 -> (F.neg ((F.exp (x259[x289_0][x289_1][x290_0] F.+ (F.neg x262[x289_0][x289_1]))) F.* x267[x289_0][x289_1][x290_0]))))))
in (let x269 = (imap2 4 16 (\x291_0 x291_1 -> (imaximum1 16 (\x292_0 -> x259[x291_0][x291_1][x292_0]))))
in (let x270 = (imap2 4 16 (\x293_0 x293_1 -> (one F./ ((isum1 16 (\x294_0 -> one)) F.+ (isum1 16 (\x295_0 -> (F.neg (indicatorp (F.neg (x259[x293_0][x293_1][x295_0] F.+ (F.neg x269[x293_0][x293_1])))))))))))
in (let x271 = (imap3 4 16 16 (\x296_0 x296_1 x296_2 -> (((F.exp (x259[x296_0][x296_1][x296_2] F.+ (F.neg x262[x296_0][x296_1]))) F.* x267[x296_0][x296_1][x296_2]) F.+ ((x268[x296_0][x296_1] F.* (one F.+ (F.neg (indicatorp (F.neg (x259[x296_0][x296_1][x296_2] F.+ (F.neg x269[x296_0][x296_1]))))))) F.* x270[x296_0][x296_1]))))
in (imap2 16 16 (\x273_0 x273_1 -> (isum1 4 (\x272_0 -> x271[x272_0][x273_0][x273_1]))))))))))))))))))
let dwpe = (let x297 = (imap1 16 (\x303_0 -> ((isum1 16 (\x304_0 -> ((wpe[x303_0][x304_0] F.+ wseq[x303_0][x304_0]) F.* (wpe[x303_0][x304_0] F.+ wseq[x303_0][x304_0])))) F./ fromi64 16)))
in (let x298 = (imap1 16 (\x305_0 -> (F.sqrt (x297[x305_0] F.+ (one F./ fromi64 100000)))))
in (let x299 = (imap2 16 16 (\x306_0 x306_1 -> x241[x306_0][x306_1]))
in (let x300 = (imap1 16 (\x307_0 -> (isum1 16 (\x308_0 -> (F.neg ((x299[x307_0][x308_0] F.* (wpe[x307_0][x308_0] F.+ wseq[x307_0][x308_0])) F.* (one F./ (x298[x307_0] F.* x298[x307_0]))))))))
in (let x301 = (imap1 16 (\x309_0 -> (x300[x309_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x297[x309_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x302_0 x302_1 -> ((x299[x302_0][x302_1] F.* (one F./ x298[x302_0])) F.+ (((x301[x302_0] F./ fromi64 16) F.* (wpe[x302_0][x302_1] F.+ wseq[x302_0][x302_1])) F.+ ((x301[x302_0] F./ fromi64 16) F.* (wpe[x302_0][x302_1] F.+ wseq[x302_0][x302_1])))))))))))
let dwqry = (imap2 16 16 (\x310_0 x310_1 -> (isum1 16 (\x311_0 -> (x199[x311_0][x310_0] F.* x9[x311_0][x310_1])))))
let dwkey = (imap2 16 16 (\x312_0 x312_1 -> (isum1 16 (\x313_0 -> (x157[x313_0][x312_0] F.* x9[x313_0][x312_1])))))
let dwval = (imap2 16 16 (\x314_0 x314_1 -> (isum1 16 (\x315_0 -> (x135[x315_0][x314_0] F.* x9[x315_0][x314_1])))))
let dwout = (imap2 16 16 (\x316_0 x316_1 -> (isum1 16 (\x317_0 -> (x117[x317_0][x316_0] F.* x27[x317_0][x316_1])))))
let dwup = (imap2 64 16 (\x318_0 x318_1 -> (isum1 16 (\x319_0 -> (x114[x319_0][x318_0] F.* x52[x319_0][x318_1])))))
let dwdown = (imap2 16 64 (\x320_0 x320_1 -> (isum1 16 (\x321_0 -> (x111[x321_0][x320_0] F.* x64[x321_0][x320_1])))))
let dwvoc = (imap2 27 16 (\x322_0 x322_1 -> (isum1 16 (\x323_0 -> (x74[x323_0][x322_0] F.* x66[x323_0][x322_1])))))
let dwseq = (let x324 = (imap1 16 (\x330_0 -> ((isum1 16 (\x331_0 -> ((wpe[x330_0][x331_0] F.+ wseq[x330_0][x331_0]) F.* (wpe[x330_0][x331_0] F.+ wseq[x330_0][x331_0])))) F./ fromi64 16)))
in (let x325 = (imap1 16 (\x332_0 -> (F.sqrt (x324[x332_0] F.+ (one F./ fromi64 100000)))))
in (let x326 = (imap2 16 16 (\x333_0 x333_1 -> x241[x333_0][x333_1]))
in (let x327 = (imap1 16 (\x334_0 -> (isum1 16 (\x335_0 -> (F.neg ((x326[x334_0][x335_0] F.* (wpe[x334_0][x335_0] F.+ wseq[x334_0][x335_0])) F.* (one F./ (x325[x334_0] F.* x325[x334_0]))))))))
in (let x328 = (imap1 16 (\x336_0 -> (x327[x336_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x324[x336_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x329_0 x329_1 -> ((x326[x329_0][x329_1] F.* (one F./ x325[x329_0])) F.+ (((x328[x329_0] F./ fromi64 16) F.* (wpe[x329_0][x329_1] F.+ wseq[x329_0][x329_1])) F.+ ((x328[x329_0] F./ fromi64 16) F.* (wpe[x329_0][x329_1] F.+ wseq[x329_0][x329_1])))))))))))
let dtarget = (imap2 16 27 (\x337_0 x337_1 -> (let x338 = (imaximum1 27 (\x342_0 -> x69[x337_0][x342_0]))
in (let x339 = (imap1 27 (\x343_0 -> (F.exp (x69[x337_0][x343_0] F.+ (F.neg x338)))))
in (let x340 = (isum1 27 (\x344_0 -> x339[x344_0]))
in (let x341 = (imap1 27 (\x345_0 -> (x339[x345_0] F.* (one F./ x340))))
in ((F.neg x72[x337_0]) F.* (F.log x341[x337_1]))))))))


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