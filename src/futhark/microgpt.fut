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
let x27 = (imap2 16 16 (\x28_0 x28_1 -> (let x29 = (imap3 4 16 16 (\x37_0 x37_1 x37_2 -> (isum1 4 (\x38_0 -> (x18[x37_0][x37_1][x38_0] F.* x21[x37_0][x37_2][x38_0])))))
in (let x30 = (imap3 4 16 16 (\x39_0 x39_1 x39_2 -> ((x29[x39_0][x39_1][x39_2] F./ fromi64 2) F.+ mask[x39_1][x39_2])))
in (let x31 = (imap2 4 16 (\x40_0 x40_1 -> (imaximum1 16 (\x41_0 -> x30[x40_0][x40_1][x41_0]))))
in (let x32 = (imap3 4 16 16 (\x42_0 x42_1 x42_2 -> (F.exp (x30[x42_0][x42_1][x42_2] F.+ (F.neg x31[x42_0][x42_1])))))
in (let x33 = (imap2 4 16 (\x43_0 x43_1 -> (isum1 16 (\x44_0 -> x32[x43_0][x43_1][x44_0]))))
in (let x34 = (imap3 4 16 16 (\x45_0 x45_1 x45_2 -> (x32[x45_0][x45_1][x45_2] F.* (one F./ x33[x45_0][x45_1]))))
in (let x35 = (imap3 4 16 16 (\x46_0 x46_1 x46_2 -> x34[x46_0][x46_1][x46_2]))
in (let x36 = (imap3 4 16 4 (\x47_0 x47_1 x47_2 -> (isum1 16 (\x48_0 -> (x35[x47_0][x47_1][x48_0] F.* x24[x47_0][x48_0][x47_2])))))
in x36[(x28_1 / 4)][x28_0][(x28_1 % 4)]))))))))))
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
let x74 = (let x75 = (imap2 16 27 (\x87_0 x87_1 -> (imaximum1 27 (\x88_0 -> x69[x87_0][x88_0]))))
in (let x76 = (imap3 16 27 27 (\x89_0 x89_1 x89_2 -> (F.exp (x69[x89_0][x89_2] F.+ (F.neg x75[x89_0][x89_1])))))
in (let x77 = (imap2 16 27 (\x90_0 x90_1 -> (isum1 27 (\x91_0 -> x76[x90_0][x90_1][x91_0]))))
in (let x78 = (imap3 16 27 27 (\x92_0 x92_1 x92_2 -> (if ((x92_2 == x92_1)) then (let x93 = (imaximum1 27 (\x97_0 -> x69[x92_0][x97_0]))
in (let x94 = (imap1 27 (\x98_0 -> (F.exp (x69[x92_0][x98_0] F.+ (F.neg x93)))))
in (let x95 = (isum1 27 (\x99_0 -> x94[x99_0]))
in (let x96 = (imap1 27 (\x100_0 -> (x94[x100_0] F.* (one F./ x95))))
in (((F.neg x72[x92_0]) F.* target[x92_0][x92_1]) F.* (one F./ x96[x92_2])))))) else zero)))
in (let x79 = (imap2 16 27 (\x101_0 x101_1 -> (isum1 27 (\x102_0 -> (F.neg ((x78[x101_0][x101_1][x102_0] F.* x76[x101_0][x101_1][x102_0]) F.* (one F./ (x77[x101_0][x101_1] F.* x77[x101_0][x101_1]))))))))
in (let x80 = (imap3 16 27 27 (\x103_0 x103_1 x103_2 -> ((x78[x103_0][x103_1][x103_2] F.* (one F./ x77[x103_0][x103_1])) F.+ x79[x103_0][x103_1])))
in (let x81 = (imap2 16 27 (\x104_0 x104_1 -> (isum1 27 (\x105_0 -> (F.neg ((F.exp (x69[x104_0][x105_0] F.+ (F.neg x75[x104_0][x104_1]))) F.* x80[x104_0][x104_1][x105_0]))))))
in (let x82 = (imap2 16 27 (\x106_0 x106_1 -> (imaximum1 27 (\x107_0 -> x69[x106_0][x107_0]))))
in (let x83 = (imap2 16 27 (\x108_0 x108_1 -> (one F./ (isum1 27 (\x109_0 -> (one F.+ (F.neg (indicatorp (F.neg (x69[x108_0][x109_0] F.+ (F.neg x82[x108_0][x108_1])))))))))))
in (imap2 16 27 (\x86_0 x86_1 -> (isum1 16 (\x84_0 -> (isum1 27 (\x85_0 -> ((if ((x86_0 == x84_0)) then ((F.exp (x69[x84_0][x86_1] F.+ (F.neg x75[x84_0][x85_0]))) F.* x80[x84_0][x85_0][x86_1]) else zero) F.+ (if ((x86_0 == x84_0)) then ((x81[x84_0][x85_0] F.* (one F.+ (F.neg (indicatorp (F.neg (x69[x84_0][x86_1] F.+ (F.neg x82[x84_0][x85_0]))))))) F.* x83[x84_0][x85_0]) else zero)))))))))))))))))
let x110 = (imap2 16 16 (\x111_0 x111_1 -> (isum1 27 (\x112_0 -> (x74[x111_0][x112_0] F.* wvoc[x112_0][x111_1])))))
let x113 = (imap2 16 64 (\x114_0 x114_1 -> ((indicatorp x61[x114_0][x114_1]) F.* (isum1 16 (\x115_0 -> (x110[x114_0][x115_0] F.* wdown[x115_0][x114_1]))))))
let x116 = (let x117 = (imap1 16 (\x125_0 -> ((isum1 16 (\x126_0 -> (x49[x125_0][x126_0] F.* x49[x125_0][x126_0]))) F./ fromi64 16)))
in (let x118 = (imap1 16 (\x127_0 -> (F.sqrt (x117[x127_0] F.+ (one F./ fromi64 100000)))))
in (let x119 = (imap2 16 16 (\x128_0 x128_1 -> (isum1 64 (\x129_0 -> (x113[x128_0][x129_0] F.* wup[x129_0][x128_1])))))
in (let x120 = (imap1 16 (\x130_0 -> (isum1 16 (\x131_0 -> (F.neg ((x119[x130_0][x131_0] F.* x49[x130_0][x131_0]) F.* (one F./ (x118[x130_0] F.* x118[x130_0]))))))))
in (let x121 = (imap1 16 (\x132_0 -> (x120[x132_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x117[x132_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x124_0 x124_1 -> (x110[x124_0][x124_1] F.+ (isum1 16 (\x122_0 -> ((if ((x124_0 == x122_0)) then (x119[x122_0][x124_1] F.* (one F./ x118[x122_0])) else zero) F.+ (isum1 16 (\x123_0 -> ((if ((x124_0 == x122_0)) then (if ((x124_1 == x123_0)) then ((x121[x122_0] F./ fromi64 16) F.* x49[x122_0][x124_1]) else zero) else zero) F.+ (if ((x124_0 == x122_0)) then (if ((x124_1 == x123_0)) then ((x121[x122_0] F./ fromi64 16) F.* x49[x122_0][x124_1]) else zero) else zero)))))))))))))))
let x133 = (imap3 4 16 4 (\x134_0 x134_1 x134_2 -> (isum1 16 (\x135_0 -> (x116[x134_1][x135_0] F.* wout[x135_0][((x134_0 * 4) + x134_2)])))))
let x136 = (imap2 16 16 (\x137_0 x137_1 -> (let x138 = (imap3 4 16 16 (\x146_0 x146_1 x146_2 -> (isum1 4 (\x147_0 -> (x18[x146_0][x146_1][x147_0] F.* x21[x146_0][x146_2][x147_0])))))
in (let x139 = (imap3 4 16 16 (\x148_0 x148_1 x148_2 -> ((x138[x148_0][x148_1][x148_2] F./ fromi64 2) F.+ mask[x148_1][x148_2])))
in (let x140 = (imap2 4 16 (\x149_0 x149_1 -> (imaximum1 16 (\x150_0 -> x139[x149_0][x149_1][x150_0]))))
in (let x141 = (imap3 4 16 16 (\x151_0 x151_1 x151_2 -> (F.exp (x139[x151_0][x151_1][x151_2] F.+ (F.neg x140[x151_0][x151_1])))))
in (let x142 = (imap2 4 16 (\x152_0 x152_1 -> (isum1 16 (\x153_0 -> x141[x152_0][x152_1][x153_0]))))
in (let x143 = (imap3 4 16 16 (\x154_0 x154_1 x154_2 -> (x141[x154_0][x154_1][x154_2] F.* (one F./ x142[x154_0][x154_1]))))
in (let x144 = (imap3 4 16 16 (\x155_0 x155_1 x155_2 -> x143[x155_0][x155_1][x155_2]))
in (let x145 = (imap3 4 16 4 (\x156_0 x156_1 x156_2 -> x133[x156_0][x156_1][x156_2]))
in (isum1 16 (\x157_0 -> (x145[(x137_1 / 4)][x157_0][(x137_1 % 4)] F.* x144[(x137_1 / 4)][x157_0][x137_0])))))))))))))
let x158 = (imap2 16 16 (\x159_0 x159_1 -> (let x160 = (imap3 4 16 16 (\x175_0 x175_1 x175_2 -> (isum1 4 (\x176_0 -> (x18[x175_0][x175_1][x176_0] F.* x21[x175_0][x175_2][x176_0])))))
in (let x161 = (imap3 4 16 16 (\x177_0 x177_1 x177_2 -> ((x160[x177_0][x177_1][x177_2] F./ fromi64 2) F.+ mask[x177_1][x177_2])))
in (let x162 = (imap3 4 16 4 (\x178_0 x178_1 x178_2 -> x133[x178_0][x178_1][x178_2]))
in (let x163 = (imap3 4 16 16 (\x179_0 x179_1 x179_2 -> (isum1 4 (\x180_0 -> (x162[x179_0][x179_1][x180_0] F.* x24[x179_0][x179_2][x180_0])))))
in (let x164 = (imap2 4 16 (\x181_0 x181_1 -> (imaximum1 16 (\x182_0 -> x161[x181_0][x181_1][x182_0]))))
in (let x165 = (imap3 4 16 16 (\x183_0 x183_1 x183_2 -> (F.exp (x161[x183_0][x183_1][x183_2] F.+ (F.neg x164[x183_0][x183_1])))))
in (let x166 = (imap2 4 16 (\x184_0 x184_1 -> (isum1 16 (\x185_0 -> x165[x184_0][x184_1][x185_0]))))
in (let x167 = (imap3 4 16 16 (\x186_0 x186_1 x186_2 -> x163[x186_0][x186_1][x186_2]))
in (let x168 = (imap2 4 16 (\x187_0 x187_1 -> (isum1 16 (\x188_0 -> (F.neg ((x167[x187_0][x187_1][x188_0] F.* x165[x187_0][x187_1][x188_0]) F.* (one F./ (x166[x187_0][x187_1] F.* x166[x187_0][x187_1]))))))))
in (let x169 = (imap3 4 16 16 (\x189_0 x189_1 x189_2 -> ((x167[x189_0][x189_1][x189_2] F.* (one F./ x166[x189_0][x189_1])) F.+ x168[x189_0][x189_1])))
in (let x170 = (imap2 4 16 (\x190_0 x190_1 -> (isum1 16 (\x191_0 -> (F.neg ((F.exp (x161[x190_0][x190_1][x191_0] F.+ (F.neg x164[x190_0][x190_1]))) F.* x169[x190_0][x190_1][x191_0]))))))
in (let x171 = (imap2 4 16 (\x192_0 x192_1 -> (imaximum1 16 (\x193_0 -> x161[x192_0][x192_1][x193_0]))))
in (let x172 = (imap2 4 16 (\x194_0 x194_1 -> (one F./ (isum1 16 (\x195_0 -> (one F.+ (F.neg (indicatorp (F.neg (x161[x194_0][x194_1][x195_0] F.+ (F.neg x171[x194_0][x194_1])))))))))))
in (let x173 = (imap3 4 16 16 (\x196_0 x196_1 x196_2 -> (isum1 16 (\x197_0 -> ((if ((x196_1 == x197_0)) then ((F.exp (x161[x196_0][x197_0][x196_2] F.+ (F.neg x164[x196_0][x197_0]))) F.* x169[x196_0][x197_0][x196_2]) else zero) F.+ (if ((x196_1 == x197_0)) then ((x170[x196_0][x197_0] F.* (one F.+ (F.neg (indicatorp (F.neg (x161[x196_0][x197_0][x196_2] F.+ (F.neg x171[x196_0][x197_0]))))))) F.* x172[x196_0][x197_0]) else zero))))))
in (let x174 = (imap3 4 16 16 (\x198_0 x198_1 x198_2 -> (x173[x198_0][x198_1][x198_2] F./ fromi64 2)))
in (isum1 16 (\x199_0 -> (x174[(x159_1 / 4)][x199_0][x159_0] F.* x18[(x159_1 / 4)][x199_0][(x159_1 % 4)]))))))))))))))))))))
let x200 = (imap2 16 16 (\x201_0 x201_1 -> (let x202 = (imap3 4 16 16 (\x217_0 x217_1 x217_2 -> (isum1 4 (\x218_0 -> (x18[x217_0][x217_1][x218_0] F.* x21[x217_0][x217_2][x218_0])))))
in (let x203 = (imap3 4 16 16 (\x219_0 x219_1 x219_2 -> ((x202[x219_0][x219_1][x219_2] F./ fromi64 2) F.+ mask[x219_1][x219_2])))
in (let x204 = (imap3 4 16 4 (\x220_0 x220_1 x220_2 -> x133[x220_0][x220_1][x220_2]))
in (let x205 = (imap3 4 16 16 (\x221_0 x221_1 x221_2 -> (isum1 4 (\x222_0 -> (x204[x221_0][x221_1][x222_0] F.* x24[x221_0][x221_2][x222_0])))))
in (let x206 = (imap2 4 16 (\x223_0 x223_1 -> (imaximum1 16 (\x224_0 -> x203[x223_0][x223_1][x224_0]))))
in (let x207 = (imap3 4 16 16 (\x225_0 x225_1 x225_2 -> (F.exp (x203[x225_0][x225_1][x225_2] F.+ (F.neg x206[x225_0][x225_1])))))
in (let x208 = (imap2 4 16 (\x226_0 x226_1 -> (isum1 16 (\x227_0 -> x207[x226_0][x226_1][x227_0]))))
in (let x209 = (imap3 4 16 16 (\x228_0 x228_1 x228_2 -> x205[x228_0][x228_1][x228_2]))
in (let x210 = (imap2 4 16 (\x229_0 x229_1 -> (isum1 16 (\x230_0 -> (F.neg ((x209[x229_0][x229_1][x230_0] F.* x207[x229_0][x229_1][x230_0]) F.* (one F./ (x208[x229_0][x229_1] F.* x208[x229_0][x229_1]))))))))
in (let x211 = (imap3 4 16 16 (\x231_0 x231_1 x231_2 -> ((x209[x231_0][x231_1][x231_2] F.* (one F./ x208[x231_0][x231_1])) F.+ x210[x231_0][x231_1])))
in (let x212 = (imap2 4 16 (\x232_0 x232_1 -> (isum1 16 (\x233_0 -> (F.neg ((F.exp (x203[x232_0][x232_1][x233_0] F.+ (F.neg x206[x232_0][x232_1]))) F.* x211[x232_0][x232_1][x233_0]))))))
in (let x213 = (imap2 4 16 (\x234_0 x234_1 -> (imaximum1 16 (\x235_0 -> x203[x234_0][x234_1][x235_0]))))
in (let x214 = (imap2 4 16 (\x236_0 x236_1 -> (one F./ (isum1 16 (\x237_0 -> (one F.+ (F.neg (indicatorp (F.neg (x203[x236_0][x236_1][x237_0] F.+ (F.neg x213[x236_0][x236_1])))))))))))
in (let x215 = (imap3 4 16 16 (\x238_0 x238_1 x238_2 -> (isum1 16 (\x239_0 -> ((if ((x238_1 == x239_0)) then ((F.exp (x203[x238_0][x239_0][x238_2] F.+ (F.neg x206[x238_0][x239_0]))) F.* x211[x238_0][x239_0][x238_2]) else zero) F.+ (if ((x238_1 == x239_0)) then ((x212[x238_0][x239_0] F.* (one F.+ (F.neg (indicatorp (F.neg (x203[x238_0][x239_0][x238_2] F.+ (F.neg x213[x238_0][x239_0]))))))) F.* x214[x238_0][x239_0]) else zero))))))
in (let x216 = (imap3 4 16 16 (\x240_0 x240_1 x240_2 -> (x215[x240_0][x240_1][x240_2] F./ fromi64 2)))
in (isum1 16 (\x241_0 -> (x216[(x201_1 / 4)][x201_0][x241_0] F.* x21[(x201_1 / 4)][x241_0][(x201_1 % 4)]))))))))))))))))))))
let x242 = (let x243 = (imap1 16 (\x251_0 -> ((isum1 16 (\x252_0 -> (x0[x251_0][x252_0] F.* x0[x251_0][x252_0]))) F./ fromi64 16)))
in (let x244 = (imap1 16 (\x253_0 -> (F.sqrt (x243[x253_0] F.+ (one F./ fromi64 100000)))))
in (let x245 = (imap2 16 16 (\x254_0 x254_1 -> (((isum1 16 (\x255_0 -> (x136[x254_0][x255_0] F.* wval[x255_0][x254_1]))) F.+ (isum1 16 (\x256_0 -> (x158[x254_0][x256_0] F.* wkey[x256_0][x254_1])))) F.+ (isum1 16 (\x257_0 -> (x200[x254_0][x257_0] F.* wqry[x257_0][x254_1]))))))
in (let x246 = (imap1 16 (\x258_0 -> (isum1 16 (\x259_0 -> (F.neg ((x245[x258_0][x259_0] F.* x0[x258_0][x259_0]) F.* (one F./ (x244[x258_0] F.* x244[x258_0]))))))))
in (let x247 = (imap1 16 (\x260_0 -> (x246[x260_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x243[x260_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x250_0 x250_1 -> (x116[x250_0][x250_1] F.+ (isum1 16 (\x248_0 -> ((if ((x250_0 == x248_0)) then (x245[x248_0][x250_1] F.* (one F./ x244[x248_0])) else zero) F.+ (isum1 16 (\x249_0 -> ((if ((x250_0 == x248_0)) then (if ((x250_1 == x249_0)) then ((x247[x248_0] F./ fromi64 16) F.* x0[x248_0][x250_1]) else zero) else zero) F.+ (if ((x250_0 == x248_0)) then (if ((x250_1 == x249_0)) then ((x247[x248_0] F./ fromi64 16) F.* x0[x248_0][x250_1]) else zero) else zero)))))))))))))))

let dmask = (let x261 = (imap3 4 16 16 (\x277_0 x277_1 x277_2 -> (isum1 4 (\x278_0 -> (x18[x277_0][x277_1][x278_0] F.* x21[x277_0][x277_2][x278_0])))))
in (let x262 = (imap3 4 16 16 (\x279_0 x279_1 x279_2 -> ((x261[x279_0][x279_1][x279_2] F./ fromi64 2) F.+ mask[x279_1][x279_2])))
in (let x263 = (imap3 4 16 4 (\x280_0 x280_1 x280_2 -> x133[x280_0][x280_1][x280_2]))
in (let x264 = (imap3 4 16 16 (\x281_0 x281_1 x281_2 -> (isum1 4 (\x282_0 -> (x263[x281_0][x281_1][x282_0] F.* x24[x281_0][x281_2][x282_0])))))
in (let x265 = (imap2 4 16 (\x283_0 x283_1 -> (imaximum1 16 (\x284_0 -> x262[x283_0][x283_1][x284_0]))))
in (let x266 = (imap3 4 16 16 (\x285_0 x285_1 x285_2 -> (F.exp (x262[x285_0][x285_1][x285_2] F.+ (F.neg x265[x285_0][x285_1])))))
in (let x267 = (imap2 4 16 (\x286_0 x286_1 -> (isum1 16 (\x287_0 -> x266[x286_0][x286_1][x287_0]))))
in (let x268 = (imap3 4 16 16 (\x288_0 x288_1 x288_2 -> x264[x288_0][x288_1][x288_2]))
in (let x269 = (imap2 4 16 (\x289_0 x289_1 -> (isum1 16 (\x290_0 -> (F.neg ((x268[x289_0][x289_1][x290_0] F.* x266[x289_0][x289_1][x290_0]) F.* (one F./ (x267[x289_0][x289_1] F.* x267[x289_0][x289_1]))))))))
in (let x270 = (imap3 4 16 16 (\x291_0 x291_1 x291_2 -> ((x268[x291_0][x291_1][x291_2] F.* (one F./ x267[x291_0][x291_1])) F.+ x269[x291_0][x291_1])))
in (let x271 = (imap2 4 16 (\x292_0 x292_1 -> (isum1 16 (\x293_0 -> (F.neg ((F.exp (x262[x292_0][x292_1][x293_0] F.+ (F.neg x265[x292_0][x292_1]))) F.* x270[x292_0][x292_1][x293_0]))))))
in (let x272 = (imap2 4 16 (\x294_0 x294_1 -> (imaximum1 16 (\x295_0 -> x262[x294_0][x294_1][x295_0]))))
in (let x273 = (imap2 4 16 (\x296_0 x296_1 -> (one F./ (isum1 16 (\x297_0 -> (one F.+ (F.neg (indicatorp (F.neg (x262[x296_0][x296_1][x297_0] F.+ (F.neg x272[x296_0][x296_1])))))))))))
in (let x274 = (imap3 4 16 16 (\x298_0 x298_1 x298_2 -> (isum1 16 (\x299_0 -> ((if ((x298_1 == x299_0)) then ((F.exp (x262[x298_0][x299_0][x298_2] F.+ (F.neg x265[x298_0][x299_0]))) F.* x270[x298_0][x299_0][x298_2]) else zero) F.+ (if ((x298_1 == x299_0)) then ((x271[x298_0][x299_0] F.* (one F.+ (F.neg (indicatorp (F.neg (x262[x298_0][x299_0][x298_2] F.+ (F.neg x272[x298_0][x299_0]))))))) F.* x273[x298_0][x299_0]) else zero))))))
in (imap2 16 16 (\x276_0 x276_1 -> (isum1 4 (\x275_0 -> x274[x275_0][x276_0][x276_1]))))))))))))))))))
let dwpe = (let x300 = (imap1 16 (\x308_0 -> ((isum1 16 (\x309_0 -> ((wpe[x308_0][x309_0] F.+ wseq[x308_0][x309_0]) F.* (wpe[x308_0][x309_0] F.+ wseq[x308_0][x309_0])))) F./ fromi64 16)))
in (let x301 = (imap1 16 (\x310_0 -> (F.sqrt (x300[x310_0] F.+ (one F./ fromi64 100000)))))
in (let x302 = (imap2 16 16 (\x311_0 x311_1 -> x242[x311_0][x311_1]))
in (let x303 = (imap1 16 (\x312_0 -> (isum1 16 (\x313_0 -> (F.neg ((x302[x312_0][x313_0] F.* (wpe[x312_0][x313_0] F.+ wseq[x312_0][x313_0])) F.* (one F./ (x301[x312_0] F.* x301[x312_0]))))))))
in (let x304 = (imap1 16 (\x314_0 -> (x303[x314_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x300[x314_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x307_0 x307_1 -> (isum1 16 (\x305_0 -> ((if ((x307_0 == x305_0)) then (x302[x305_0][x307_1] F.* (one F./ x301[x305_0])) else zero) F.+ (isum1 16 (\x306_0 -> ((if ((x307_0 == x305_0)) then (if ((x307_1 == x306_0)) then ((x304[x305_0] F./ fromi64 16) F.* (wpe[x305_0][x307_1] F.+ wseq[x305_0][x307_1])) else zero) else zero) F.+ (if ((x307_0 == x305_0)) then (if ((x307_1 == x306_0)) then ((x304[x305_0] F./ fromi64 16) F.* (wpe[x305_0][x307_1] F.+ wseq[x305_0][x307_1])) else zero) else zero))))))))))))))
let dwqry = (imap2 16 16 (\x315_0 x315_1 -> (isum1 16 (\x316_0 -> (x200[x316_0][x315_0] F.* x9[x316_0][x315_1])))))
let dwkey = (imap2 16 16 (\x317_0 x317_1 -> (isum1 16 (\x318_0 -> (x158[x318_0][x317_0] F.* x9[x318_0][x317_1])))))
let dwval = (imap2 16 16 (\x319_0 x319_1 -> (isum1 16 (\x320_0 -> (x136[x320_0][x319_0] F.* x9[x320_0][x319_1])))))
let dwout = (imap2 16 16 (\x321_0 x321_1 -> (isum1 16 (\x322_0 -> (x116[x322_0][x321_0] F.* x27[x322_0][x321_1])))))
let dwup = (imap2 64 16 (\x323_0 x323_1 -> (isum1 16 (\x324_0 -> (x113[x324_0][x323_0] F.* x52[x324_0][x323_1])))))
let dwdown = (imap2 16 64 (\x325_0 x325_1 -> (isum1 16 (\x326_0 -> (x110[x326_0][x325_0] F.* x64[x326_0][x325_1])))))
let dwvoc = (imap2 27 16 (\x327_0 x327_1 -> (isum1 16 (\x328_0 -> (x74[x328_0][x327_0] F.* x66[x328_0][x327_1])))))
let dwseq = (let x329 = (imap1 16 (\x337_0 -> ((isum1 16 (\x338_0 -> ((wpe[x337_0][x338_0] F.+ wseq[x337_0][x338_0]) F.* (wpe[x337_0][x338_0] F.+ wseq[x337_0][x338_0])))) F./ fromi64 16)))
in (let x330 = (imap1 16 (\x339_0 -> (F.sqrt (x329[x339_0] F.+ (one F./ fromi64 100000)))))
in (let x331 = (imap2 16 16 (\x340_0 x340_1 -> x242[x340_0][x340_1]))
in (let x332 = (imap1 16 (\x341_0 -> (isum1 16 (\x342_0 -> (F.neg ((x331[x341_0][x342_0] F.* (wpe[x341_0][x342_0] F.+ wseq[x341_0][x342_0])) F.* (one F./ (x330[x341_0] F.* x330[x341_0]))))))))
in (let x333 = (imap1 16 (\x343_0 -> (x332[x343_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x329[x343_0] F.+ (one F./ fromi64 100000))))))))
in (imap2 16 16 (\x336_0 x336_1 -> (isum1 16 (\x334_0 -> ((if ((x336_0 == x334_0)) then (x331[x334_0][x336_1] F.* (one F./ x330[x334_0])) else zero) F.+ (isum1 16 (\x335_0 -> ((if ((x336_0 == x334_0)) then (if ((x336_1 == x335_0)) then ((x333[x334_0] F./ fromi64 16) F.* (wpe[x334_0][x336_1] F.+ wseq[x334_0][x336_1])) else zero) else zero) F.+ (if ((x336_0 == x334_0)) then (if ((x336_1 == x335_0)) then ((x333[x334_0] F./ fromi64 16) F.* (wpe[x334_0][x336_1] F.+ wseq[x334_0][x336_1])) else zero) else zero))))))))))))))
let dtarget = (imap2 16 27 (\x344_0 x344_1 -> (let x345 = (imaximum1 27 (\x349_0 -> x69[x344_0][x349_0]))
in (let x346 = (imap1 27 (\x350_0 -> (F.exp (x69[x344_0][x350_0] F.+ (F.neg x345)))))
in (let x347 = (isum1 27 (\x351_0 -> x346[x351_0]))
in (let x348 = (imap1 27 (\x352_0 -> (x346[x352_0] F.* (one F./ x347))))
in ((F.neg x72[x344_0]) F.* (F.log x348[x344_1]))))))))

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

entry forward_seq (p : params [16]) (tokens : [16]i64) (mask : [16][16]f64) : [16][27]f64 =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let wseq = (imap2 16 16 (\m n -> wte[tokens[m]][n]))
   in nn64.forward_seq mask wpe wqry wkey wval wout wup wdown wvoc wseq

entry cal_loss (p : params [16]) (tokens : [16]i64) (target : [16][27]f64) (mask : [16][16]f64) : (f64 , [16]f64) =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let wseq = (imap2 16 16 (\m n -> wte[tokens[m]][n]))
   in nn64.cal_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target

entry grad_loss (p : params [16]) (tokens : [16]i64) (target : [16][27]f64) (mask : [16][16]f64) :
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
   let wseq = (imap2 16 16 (\m n -> wte[tokens[m]][n]))
   let (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq) =
    nn64.grad_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target
   let dwte = (imap2 27 16 (\m n -> nn64.isum1 16 (\k -> if (tokens[k] == m) then dwseq[k][n] else nn64.zero)))
   in  (dwte, dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc)
