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

let x0 = (imap2 16 16 (\x1_0 x1_1 -> ((wpe[x1_0][x1_1] F.+ wseq[x1_0][x1_1]) F.* (one F./ (F.sqrt (((isum1 16 (\x2_0 -> ((wpe[x1_0][x2_0] F.+ wseq[x1_0][x2_0]) F.* (wpe[x1_0][x2_0] F.+ wseq[x1_0][x2_0])))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))))))
let x3 = (imap2 16 16 (\x4_0 x4_1 -> (x0[x4_0][x4_1] F.* (one F./ (F.sqrt (((isum1 16 (\x5_0 -> (x0[x4_0][x5_0] F.* x0[x4_0][x5_0]))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))))))
let x6 = (imap2 16 16 (\x7_0 x7_1 -> (isum1 16 (\x8_0 -> (wqry[x7_1][x8_0] F.* x3[x7_0][x8_0])))))
let x9 = (imap2 16 16 (\x10_0 x10_1 -> (isum1 16 (\x11_0 -> (wkey[x10_1][x11_0] F.* x3[x10_0][x11_0])))))
let x12 = (imap2 16 16 (\x13_0 x13_1 -> (isum1 16 (\x14_0 -> (wval[x13_1][x14_0] F.* x3[x13_0][x14_0])))))
let x15 = (imap3 4 16 4 (\x16_0 x16_1 x16_2 -> x6[x16_1][((x16_0 * 4) + x16_2)]))
let x17 = (imap3 4 16 4 (\x18_0 x18_1 x18_2 -> x9[x18_1][((x18_0 * 4) + x18_2)]))
let x19 = (imap3 4 16 4 (\x20_0 x20_1 x20_2 -> x12[x20_1][((x20_0 * 4) + x20_2)]))
let x21 = (imap3 4 16 4 (\x22_0 x22_1 x22_2 -> (let x23 = (imap2 16 16 (\x24_0 x24_1 -> (((isum1 4 (\x25_0 -> (x15[x22_0][x24_0][x25_0] F.* x17[x22_0][x24_1][x25_0]))) F./ fromi64 2) F.+ mask[x24_0][x24_1])))
in (isum1 16 (\x26_0 -> (let x27 = (imap1 16 (\x28_0 -> (F.exp (x23[x22_1][x28_0] F.+ (F.neg (imaximum1 16 (\x29_0 -> x23[x22_1][x29_0])))))))
in ((x27[x26_0] F.* (one F./ (isum1 16 (\x30_0 -> x27[x30_0])))) F.* x19[x22_0][x26_0][x22_2])))))))
let x31 = (imap2 16 16 (\x32_0 x32_1 -> x21[(x32_1 / 4)][x32_0][(x32_1 % 4)]))
let x33 = (imap2 16 16 (\x34_0 x34_1 -> (isum1 16 (\x35_0 -> (wout[x34_1][x35_0] F.* x31[x34_0][x35_0])))))
let x36 = (imap2 16 16 (\x37_0 x37_1 -> (x33[x37_0][x37_1] F.+ x0[x37_0][x37_1])))
let x38 = (imap2 16 16 (\x39_0 x39_1 -> (x36[x39_0][x39_1] F.* (one F./ (F.sqrt (((isum1 16 (\x40_0 -> (x36[x39_0][x40_0] F.* x36[x39_0][x40_0]))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))))))
let x41 = (imap2 16 64 (\x42_0 x42_1 -> (isum1 16 (\x43_0 -> (wup[x42_1][x43_0] F.* x38[x42_0][x43_0])))))
let x44 = (imap2 16 64 (\x45_0 x45_1 -> F.max x41[x45_0][x45_1] zero))
let x46 = (imap2 16 16 (\x47_0 x47_1 -> (isum1 64 (\x48_0 -> (wdown[x47_1][x48_0] F.* x44[x47_0][x48_0])))))
let x49 = (imap2 16 16 (\x50_0 x50_1 -> (x46[x50_0][x50_1] F.+ x36[x50_0][x50_1])))
let x51 = (imap2 16 27 (\x52_0 x52_1 -> (isum1 16 (\x53_0 -> (wvoc[x52_1][x53_0] F.* x49[x52_0][x53_0])))))
let x54 = one
let x55 = (imap1 16 (\x56_0 -> (x54 F./ fromi64 16)))
let x57 = (let x58 = (imap2 16 27 (\x65_0 x65_1 -> (imaximum1 27 (\x66_0 -> x51[x65_0][x66_0]))))
in (let x59 = (imap3 16 27 27 (\x67_0 x67_1 x67_2 -> (F.exp (x51[x67_0][x67_2] F.+ (F.neg x58[x67_0][x67_1])))))
in (let x60 = (imap2 16 27 (\x68_0 x68_1 -> (isum1 27 (\x69_0 -> x59[x68_0][x68_1][x69_0]))))
in (let x61 = (imap3 16 27 27 (\x70_0 x70_1 x70_2 -> (if ((x70_2 == x70_1)) then (let x71 = (imap1 27 (\x72_0 -> (F.exp (x51[x70_0][x72_0] F.+ (F.neg (imaximum1 27 (\x73_0 -> x51[x70_0][x73_0])))))))
in (((F.neg x55[x70_0]) F.* target[x70_0][x70_1]) F.* (one F./ (x71[x70_2] F.* (one F./ (isum1 27 (\x74_0 -> x71[x74_0]))))))) else zero)))
in (let x62 = (imap3 16 27 27 (\x75_0 x75_1 x75_2 -> ((x61[x75_0][x75_1][x75_2] F.* (one F./ x60[x75_0][x75_1])) F.+ (isum1 27 (\x76_0 -> (F.neg ((x61[x75_0][x75_1][x76_0] F.* x59[x75_0][x75_1][x76_0]) F.* (one F./ (x60[x75_0][x75_1] F.* x60[x75_0][x75_1])))))))))
in (let x63 = (imap2 16 27 (\x77_0 x77_1 -> (imaximum1 27 (\x78_0 -> x51[x77_0][x78_0]))))
in (imap2 16 27 (\x64_0 x64_1 -> (isum1 27 (\x79_0 -> (((F.exp (x51[x64_0][x64_1] F.+ (F.neg x58[x64_0][x79_0]))) F.* x62[x64_0][x79_0][x64_1]) F.+ (((isum1 27 (\x80_0 -> (F.neg ((F.exp (x51[x64_0][x80_0] F.+ (F.neg x58[x64_0][x79_0]))) F.* x62[x64_0][x79_0][x80_0])))) F.* (x54 F.+ (F.neg (indicatorp (F.neg (x51[x64_0][x64_1] F.+ (F.neg x63[x64_0][x79_0]))))))) F.* (one F./ (isum1 27 (\x81_0 -> (x54 F.+ (F.neg (indicatorp (F.neg (x51[x64_0][x81_0] F.+ (F.neg x63[x64_0][x79_0])))))))))))))))))))))
let x82 = (imap2 16 16 (\x83_0 x83_1 -> (isum1 27 (\x84_0 -> (x57[x83_0][x84_0] F.* wvoc[x84_0][x83_1])))))
let x85 = (imap2 16 16 (\x86_0 x86_1 -> x82[x86_0][x86_1]))
let x87 = (imap2 16 64 (\x88_0 x88_1 -> (isum1 16 (\x89_0 -> (x85[x88_0][x89_0] F.* wdown[x89_0][x88_1])))))
let x90 = (imap2 16 64 (\x91_0 x91_1 -> ((indicatorp x41[x91_0][x91_1]) F.* x87[x91_0][x91_1])))
let x92 = (imap2 16 16 (\x93_0 x93_1 -> (isum1 64 (\x94_0 -> (x90[x93_0][x94_0] F.* wup[x94_0][x93_1])))))
let x95 = (let x96 = (imap1 16 (\x101_0 -> ((isum1 16 (\x102_0 -> (x36[x101_0][x102_0] F.* x36[x101_0][x102_0]))) F./ fromi64 16)))
in (let x97 = (imap1 16 (\x103_0 -> (F.sqrt (x96[x103_0] F.+ (x54 F./ fromi64 100000)))))
in (let x98 = (imap2 16 16 (\x104_0 x104_1 -> x92[x104_0][x104_1]))
in (let x99 = (imap1 16 (\x105_0 -> ((isum1 16 (\x106_0 -> (F.neg ((x98[x105_0][x106_0] F.* x36[x105_0][x106_0]) F.* (one F./ (x97[x105_0] F.* x97[x105_0])))))) F.* (one F./ ((x54 F.+ x54) F.* (F.sqrt (x96[x105_0] F.+ (x54 F./ fromi64 100000))))))))
in (imap2 16 16 (\x100_0 x100_1 -> (x85[x100_0][x100_1] F.+ ((x98[x100_0][x100_1] F.* (one F./ x97[x100_0])) F.+ (((x99[x100_0] F./ fromi64 16) F.* x36[x100_0][x100_1]) F.+ ((x99[x100_0] F./ fromi64 16) F.* x36[x100_0][x100_1]))))))))))
let x107 = (imap2 16 16 (\x108_0 x108_1 -> x95[x108_0][x108_1]))
let x109 = (imap2 16 16 (\x110_0 x110_1 -> (isum1 16 (\x111_0 -> (x107[x110_0][x111_0] F.* wout[x111_0][x110_1])))))
let x112 = (imap3 4 16 4 (\x113_0 x113_1 x113_2 -> x109[x113_1][((x113_0 * 4) + x113_2)]))
let x114 = (let x115 = (imap3 4 16 16 (\x117_0 x117_1 x117_2 -> (((isum1 4 (\x118_0 -> (x15[x117_0][x117_1][x118_0] F.* x17[x117_0][x117_2][x118_0]))) F./ fromi64 2) F.+ mask[x117_1][x117_2])))
in (imap3 4 16 4 (\x116_0 x116_1 x116_2 -> (isum1 16 (\x119_0 -> (let x120 = (imap1 16 (\x121_0 -> (F.exp (x115[x116_0][x119_0][x121_0] F.+ (F.neg (imaximum1 16 (\x122_0 -> x115[x116_0][x119_0][x122_0])))))))
in (x112[x116_0][x119_0][x116_2] F.* (x120[x116_1] F.* (one F./ (isum1 16 (\x123_0 -> x120[x123_0])))))))))))
let x124 = (let x125 = (imap3 4 16 16 (\x127_0 x127_1 x127_2 -> (((isum1 4 (\x128_0 -> (x15[x127_0][x127_1][x128_0] F.* x17[x127_0][x127_2][x128_0]))) F./ fromi64 2) F.+ mask[x127_1][x127_2])))
in (imap3 4 16 4 (\x126_0 x126_1 x126_2 -> (isum1 16 (\x129_0 -> (let x130 = (imap1 16 (\x136_0 -> (imaximum1 16 (\x137_0 -> x125[x126_0][x136_0][x137_0]))))
in (let x131 = (imap2 16 16 (\x138_0 x138_1 -> (F.exp (x125[x126_0][x138_0][x138_1] F.+ (F.neg x130[x138_0])))))
in (let x132 = (imap1 16 (\x139_0 -> (isum1 16 (\x140_0 -> x131[x139_0][x140_0]))))
in (let x133 = (imap2 16 16 (\x141_0 x141_1 -> (isum1 4 (\x142_0 -> (x112[x126_0][x141_0][x142_0] F.* x19[x126_0][x141_1][x142_0])))))
in (let x134 = (imap2 16 16 (\x143_0 x143_1 -> ((x133[x143_0][x143_1] F.* (one F./ x132[x143_0])) F.+ (isum1 16 (\x144_0 -> (F.neg ((x133[x143_0][x144_0] F.* x131[x143_0][x144_0]) F.* (one F./ (x132[x143_0] F.* x132[x143_0])))))))))
in (let x135 = (imap1 16 (\x145_0 -> (imaximum1 16 (\x146_0 -> x125[x126_0][x145_0][x146_0]))))
in (((((F.exp (x125[x126_0][x129_0][x126_1] F.+ (F.neg x130[x129_0]))) F.* x134[x129_0][x126_1]) F.+ (((isum1 16 (\x147_0 -> (F.neg ((F.exp (x125[x126_0][x129_0][x147_0] F.+ (F.neg x130[x129_0]))) F.* x134[x129_0][x147_0])))) F.* (x54 F.+ (F.neg (indicatorp (F.neg (x125[x126_0][x129_0][x126_1] F.+ (F.neg x135[x129_0]))))))) F.* (one F./ (isum1 16 (\x148_0 -> (x54 F.+ (F.neg (indicatorp (F.neg (x125[x126_0][x129_0][x148_0] F.+ (F.neg x135[x129_0]))))))))))) F./ fromi64 2) F.* x15[x126_0][x129_0][x126_2]))))))))))))
let x149 = (let x150 = (imap3 4 16 16 (\x152_0 x152_1 x152_2 -> (((isum1 4 (\x153_0 -> (x15[x152_0][x152_1][x153_0] F.* x17[x152_0][x152_2][x153_0]))) F./ fromi64 2) F.+ mask[x152_1][x152_2])))
in (imap3 4 16 4 (\x151_0 x151_1 x151_2 -> (isum1 16 (\x154_0 -> (let x155 = (imap1 16 (\x161_0 -> (imaximum1 16 (\x162_0 -> x150[x151_0][x161_0][x162_0]))))
in (let x156 = (imap2 16 16 (\x163_0 x163_1 -> (F.exp (x150[x151_0][x163_0][x163_1] F.+ (F.neg x155[x163_0])))))
in (let x157 = (imap1 16 (\x164_0 -> (isum1 16 (\x165_0 -> x156[x164_0][x165_0]))))
in (let x158 = (imap2 16 16 (\x166_0 x166_1 -> (isum1 4 (\x167_0 -> (x112[x151_0][x166_0][x167_0] F.* x19[x151_0][x166_1][x167_0])))))
in (let x159 = (imap2 16 16 (\x168_0 x168_1 -> ((x158[x168_0][x168_1] F.* (one F./ x157[x168_0])) F.+ (isum1 16 (\x169_0 -> (F.neg ((x158[x168_0][x169_0] F.* x156[x168_0][x169_0]) F.* (one F./ (x157[x168_0] F.* x157[x168_0])))))))))
in (let x160 = (imap1 16 (\x170_0 -> (imaximum1 16 (\x171_0 -> x150[x151_0][x170_0][x171_0]))))
in (((((F.exp (x150[x151_0][x151_1][x154_0] F.+ (F.neg x155[x151_1]))) F.* x159[x151_1][x154_0]) F.+ (((isum1 16 (\x172_0 -> (F.neg ((F.exp (x150[x151_0][x151_1][x172_0] F.+ (F.neg x155[x151_1]))) F.* x159[x151_1][x172_0])))) F.* (x54 F.+ (F.neg (indicatorp (F.neg (x150[x151_0][x151_1][x154_0] F.+ (F.neg x160[x151_1]))))))) F.* (one F./ (isum1 16 (\x173_0 -> (x54 F.+ (F.neg (indicatorp (F.neg (x150[x151_0][x151_1][x173_0] F.+ (F.neg x160[x151_1]))))))))))) F./ fromi64 2) F.* x17[x151_0][x154_0][x151_2]))))))))))))
let x174 = (imap2 16 16 (\x175_0 x175_1 -> x114[(x175_1 / 4)][x175_0][(x175_1 % 4)]))
let x176 = (imap2 16 16 (\x177_0 x177_1 -> x124[(x177_1 / 4)][x177_0][(x177_1 % 4)]))
let x178 = (imap2 16 16 (\x179_0 x179_1 -> x149[(x179_1 / 4)][x179_0][(x179_1 % 4)]))
let x180 = (imap2 16 16 (\x181_0 x181_1 -> (((isum1 16 (\x182_0 -> (x174[x181_0][x182_0] F.* wval[x182_0][x181_1]))) F.+ (isum1 16 (\x183_0 -> (x176[x181_0][x183_0] F.* wkey[x183_0][x181_1])))) F.+ (isum1 16 (\x184_0 -> (x178[x181_0][x184_0] F.* wqry[x184_0][x181_1]))))))
let x185 = (let x186 = (imap1 16 (\x191_0 -> ((isum1 16 (\x192_0 -> (x0[x191_0][x192_0] F.* x0[x191_0][x192_0]))) F./ fromi64 16)))
in (let x187 = (imap1 16 (\x193_0 -> (F.sqrt (x186[x193_0] F.+ (x54 F./ fromi64 100000)))))
in (let x188 = (imap2 16 16 (\x194_0 x194_1 -> x180[x194_0][x194_1]))
in (let x189 = (imap1 16 (\x195_0 -> ((isum1 16 (\x196_0 -> (F.neg ((x188[x195_0][x196_0] F.* x0[x195_0][x196_0]) F.* (one F./ (x187[x195_0] F.* x187[x195_0])))))) F.* (one F./ ((x54 F.+ x54) F.* (F.sqrt (x186[x195_0] F.+ (x54 F./ fromi64 100000))))))))
in (imap2 16 16 (\x190_0 x190_1 -> (x107[x190_0][x190_1] F.+ ((x188[x190_0][x190_1] F.* (one F./ x187[x190_0])) F.+ (((x189[x190_0] F./ fromi64 16) F.* x0[x190_0][x190_1]) F.+ ((x189[x190_0] F./ fromi64 16) F.* x0[x190_0][x190_1]))))))))))

let dmask = (let x197 = (imap3 4 16 16 (\x213_0 x213_1 x213_2 -> (isum1 4 (\x214_0 -> (x15[x213_0][x213_1][x214_0] F.* x17[x213_0][x213_2][x214_0])))))
in (let x198 = (imap3 4 16 16 (\x215_0 x215_1 x215_2 -> ((x197[x215_0][x215_1][x215_2] F./ fromi64 2) F.+ mask[x215_1][x215_2])))
in (let x199 = (imap3 4 16 4 (\x216_0 x216_1 x216_2 -> x112[x216_0][x216_1][x216_2]))
in (let x200 = (imap3 4 16 16 (\x217_0 x217_1 x217_2 -> (isum1 4 (\x218_0 -> (x199[x217_0][x217_1][x218_0] F.* x19[x217_0][x217_2][x218_0])))))
in (let x201 = (imap2 4 16 (\x219_0 x219_1 -> (imaximum1 16 (\x220_0 -> x198[x219_0][x219_1][x220_0]))))
in (let x202 = (imap3 4 16 16 (\x221_0 x221_1 x221_2 -> (F.exp (x198[x221_0][x221_1][x221_2] F.+ (F.neg x201[x221_0][x221_1])))))
in (let x203 = (imap2 4 16 (\x222_0 x222_1 -> (isum1 16 (\x223_0 -> x202[x222_0][x222_1][x223_0]))))
in (let x204 = (imap3 4 16 16 (\x224_0 x224_1 x224_2 -> x200[x224_0][x224_1][x224_2]))
in (let x205 = (imap2 4 16 (\x225_0 x225_1 -> (isum1 16 (\x226_0 -> (F.neg ((x204[x225_0][x225_1][x226_0] F.* x202[x225_0][x225_1][x226_0]) F.* (one F./ (x203[x225_0][x225_1] F.* x203[x225_0][x225_1]))))))))
in (let x206 = (imap3 4 16 16 (\x227_0 x227_1 x227_2 -> ((x204[x227_0][x227_1][x227_2] F.* (one F./ x203[x227_0][x227_1])) F.+ x205[x227_0][x227_1])))
in (let x207 = (imap2 4 16 (\x228_0 x228_1 -> (isum1 16 (\x229_0 -> (F.neg ((F.exp (x198[x228_0][x228_1][x229_0] F.+ (F.neg x201[x228_0][x228_1]))) F.* x206[x228_0][x228_1][x229_0]))))))
in (let x208 = (imap2 4 16 (\x230_0 x230_1 -> (imaximum1 16 (\x231_0 -> x198[x230_0][x230_1][x231_0]))))
in (let x209 = (imap2 4 16 (\x232_0 x232_1 -> (one F./ (isum1 16 (\x233_0 -> (x54 F.+ (F.neg (indicatorp (F.neg (x198[x232_0][x232_1][x233_0] F.+ (F.neg x208[x232_0][x232_1])))))))))))
in (let x210 = (imap3 4 16 16 (\x234_0 x234_1 x234_2 -> (((F.exp (x198[x234_0][x234_1][x234_2] F.+ (F.neg x201[x234_0][x234_1]))) F.* x206[x234_0][x234_1][x234_2]) F.+ ((x207[x234_0][x234_1] F.* (x54 F.+ (F.neg (indicatorp (F.neg (x198[x234_0][x234_1][x234_2] F.+ (F.neg x208[x234_0][x234_1]))))))) F.* x209[x234_0][x234_1]))))
in (imap2 16 16 (\x212_0 x212_1 -> (isum1 4 (\x211_0 -> x210[x211_0][x212_0][x212_1]))))))))))))))))))
let dwpe = (let x235 = (imap1 16 (\x241_0 -> ((isum1 16 (\x242_0 -> ((wpe[x241_0][x242_0] F.+ wseq[x241_0][x242_0]) F.* (wpe[x241_0][x242_0] F.+ wseq[x241_0][x242_0])))) F./ fromi64 16)))
in (let x236 = (imap1 16 (\x243_0 -> (F.sqrt (x235[x243_0] F.+ (x54 F./ fromi64 100000)))))
in (let x237 = (imap2 16 16 (\x244_0 x244_1 -> x185[x244_0][x244_1]))
in (let x238 = (imap1 16 (\x245_0 -> (isum1 16 (\x246_0 -> (F.neg ((x237[x245_0][x246_0] F.* (wpe[x245_0][x246_0] F.+ wseq[x245_0][x246_0])) F.* (one F./ (x236[x245_0] F.* x236[x245_0]))))))))
in (let x239 = (imap1 16 (\x247_0 -> (x238[x247_0] F.* (one F./ ((x54 F.+ x54) F.* (F.sqrt (x235[x247_0] F.+ (x54 F./ fromi64 100000))))))))
in (imap2 16 16 (\x240_0 x240_1 -> ((x237[x240_0][x240_1] F.* (one F./ x236[x240_0])) F.+ (((x239[x240_0] F./ fromi64 16) F.* (wpe[x240_0][x240_1] F.+ wseq[x240_0][x240_1])) F.+ ((x239[x240_0] F./ fromi64 16) F.* (wpe[x240_0][x240_1] F.+ wseq[x240_0][x240_1])))))))))))
let dwqry = (imap2 16 16 (\x248_0 x248_1 -> (isum1 16 (\x249_0 -> (x178[x249_0][x248_0] F.* x3[x249_0][x248_1])))))
let dwkey = (imap2 16 16 (\x250_0 x250_1 -> (isum1 16 (\x251_0 -> (x176[x251_0][x250_0] F.* x3[x251_0][x250_1])))))
let dwval = (imap2 16 16 (\x252_0 x252_1 -> (isum1 16 (\x253_0 -> (x174[x253_0][x252_0] F.* x3[x253_0][x252_1])))))
let dwout = (imap2 16 16 (\x254_0 x254_1 -> (isum1 16 (\x255_0 -> (x107[x255_0][x254_0] F.* x31[x255_0][x254_1])))))
let dwup = (imap2 64 16 (\x256_0 x256_1 -> (isum1 16 (\x257_0 -> (x90[x257_0][x256_0] F.* x38[x257_0][x256_1])))))
let dwdown = (imap2 16 64 (\x258_0 x258_1 -> (isum1 16 (\x259_0 -> (x85[x259_0][x258_0] F.* x44[x259_0][x258_1])))))
let dwvoc = (imap2 27 16 (\x260_0 x260_1 -> (isum1 16 (\x261_0 -> (x57[x261_0][x260_0] F.* x49[x261_0][x260_1])))))
let dwseq = (let x262 = (imap1 16 (\x268_0 -> ((isum1 16 (\x269_0 -> ((wpe[x268_0][x269_0] F.+ wseq[x268_0][x269_0]) F.* (wpe[x268_0][x269_0] F.+ wseq[x268_0][x269_0])))) F./ fromi64 16)))
in (let x263 = (imap1 16 (\x270_0 -> (F.sqrt (x262[x270_0] F.+ (x54 F./ fromi64 100000)))))
in (let x264 = (imap2 16 16 (\x271_0 x271_1 -> x185[x271_0][x271_1]))
in (let x265 = (imap1 16 (\x272_0 -> (isum1 16 (\x273_0 -> (F.neg ((x264[x272_0][x273_0] F.* (wpe[x272_0][x273_0] F.+ wseq[x272_0][x273_0])) F.* (one F./ (x263[x272_0] F.* x263[x272_0]))))))))
in (let x266 = (imap1 16 (\x274_0 -> (x265[x274_0] F.* (one F./ ((x54 F.+ x54) F.* (F.sqrt (x262[x274_0] F.+ (x54 F./ fromi64 100000))))))))
in (imap2 16 16 (\x267_0 x267_1 -> ((x264[x267_0][x267_1] F.* (one F./ x263[x267_0])) F.+ (((x266[x267_0] F./ fromi64 16) F.* (wpe[x267_0][x267_1] F.+ wseq[x267_0][x267_1])) F.+ ((x266[x267_0] F./ fromi64 16) F.* (wpe[x267_0][x267_1] F.+ wseq[x267_0][x267_1])))))))))))
let dtarget = (imap2 16 27 (\x275_0 x275_1 -> (let x276 = (imaximum1 27 (\x280_0 -> x51[x275_0][x280_0]))
in (let x277 = (imap1 27 (\x281_0 -> (F.exp (x51[x275_0][x281_0] F.+ (F.neg x276)))))
in (let x278 = (isum1 27 (\x282_0 -> x277[x282_0]))
in (let x279 = (imap1 27 (\x283_0 -> (x277[x283_0] F.* (one F./ x278))))
in ((F.neg x55[x275_0]) F.* (F.log x279[x275_1]))))))))

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