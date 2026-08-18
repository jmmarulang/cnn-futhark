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

let x0 = (imap2 16 16 (\x1_0 x1_1 -> (wpe[x1_0][x1_1] F.+ wseq[x1_0][x1_1])))
let x2 = (imap2 16 16 (\x3_0 x3_1 -> (let x4 = (imap1 16 (\x7_0 -> (x0[x3_0][x7_0] F.* x0[x3_0][x7_0])))
in (let x5 = ((isum1 16 (\x8_0 -> x4[x8_0])) F./ fromi64 16)
in (let x6 = (F.sqrt (x5 F.+ (one F./ fromi64 100000)))
in (x0[x3_0][x3_1] F.* (one F./ x6)))))))
let x9 = (imap2 16 16 (\x10_0 x10_1 -> (let x11 = (imap1 16 (\x14_0 -> (x2[x10_0][x14_0] F.* x2[x10_0][x14_0])))
in (let x12 = ((isum1 16 (\x15_0 -> x11[x15_0])) F./ fromi64 16)
in (let x13 = (F.sqrt (x12 F.+ (one F./ fromi64 100000)))
in (x2[x10_0][x10_1] F.* (one F./ x13)))))))
let x16 = (imap2 16 16 (\x17_0 x17_1 -> (isum1 16 (\x18_0 -> (wqry[x17_1][x18_0] F.* x9[x17_0][x18_0])))))
let x19 = (imap2 16 16 (\x20_0 x20_1 -> (isum1 16 (\x21_0 -> (wkey[x20_1][x21_0] F.* x9[x20_0][x21_0])))))
let x22 = (imap2 16 16 (\x23_0 x23_1 -> (isum1 16 (\x24_0 -> (wval[x23_1][x24_0] F.* x9[x23_0][x24_0])))))
let x25 = (imap3 4 16 4 (\x26_0 x26_1 x26_2 -> x16[x26_1][((x26_0 * 4) + x26_2)]))
let x27 = (imap3 4 16 4 (\x28_0 x28_1 x28_2 -> x19[x28_1][((x28_0 * 4) + x28_2)]))
let x29 = (imap3 4 16 4 (\x30_0 x30_1 x30_2 -> x22[x30_1][((x30_0 * 4) + x30_2)]))
let x31 = (imap3 4 16 4 (\x32_0 x32_1 x32_2 -> (let x33 = (imap2 16 16 (\x37_0 x37_1 -> (isum1 4 (\x38_0 -> (x25[x32_0][x37_0][x38_0] F.* x27[x32_0][x37_1][x38_0])))))
in (let x34 = (imap2 16 16 (\x39_0 x39_1 -> ((x33[x39_0][x39_1] F./ fromi64 2) F.+ mask[x39_0][x39_1])))
in (let x35 = (imap2 16 16 (\x40_0 x40_1 -> (imaximum1 16 (\x41_0 -> x34[x40_0][x41_0]))))
in (let x36 = (imap2 16 16 (\x42_0 x42_1 -> (let x43 = (imap1 16 (\x45_0 -> (F.exp (x34[x42_0][x45_0] F.+ (F.neg x35[x42_0][x45_0])))))
in (let x44 = (one F./ (isum1 16 (\x46_0 -> x43[x46_0])))
in (x43[x42_1] F.* x44)))))
in (isum1 16 (\x47_0 -> (x36[x32_1][x47_0] F.* x29[x32_0][x47_0][x32_2])))))))))
let x48 = (imap2 16 16 (\x49_0 x49_1 -> x31[(x49_1 / 4)][x49_0][(x49_1 % 4)]))
let x50 = (imap2 16 16 (\x51_0 x51_1 -> (isum1 16 (\x52_0 -> (wout[x51_1][x52_0] F.* x48[x51_0][x52_0])))))
let x53 = (imap2 16 16 (\x54_0 x54_1 -> (x50[x54_0][x54_1] F.+ x2[x54_0][x54_1])))
let x55 = (imap2 16 16 (\x56_0 x56_1 -> (let x57 = (imap1 16 (\x60_0 -> (x53[x56_0][x60_0] F.* x53[x56_0][x60_0])))
in (let x58 = ((isum1 16 (\x61_0 -> x57[x61_0])) F./ fromi64 16)
in (let x59 = (F.sqrt (x58 F.+ (one F./ fromi64 100000)))
in (x53[x56_0][x56_1] F.* (one F./ x59)))))))
let x62 = (imap2 16 64 (\x63_0 x63_1 -> (isum1 16 (\x64_0 -> (wup[x63_1][x64_0] F.* x55[x63_0][x64_0])))))
let x65 = (imap2 16 64 (\x66_0 x66_1 -> F.max x62[x66_0][x66_1] zero))
let x67 = (imap2 16 16 (\x68_0 x68_1 -> (isum1 64 (\x69_0 -> (wdown[x68_1][x69_0] F.* x65[x68_0][x69_0])))))
let x70 = (imap2 16 16 (\x71_0 x71_1 -> (x67[x71_0][x71_1] F.+ x53[x71_0][x71_1])))
let x72 = (imap2 16 27 (\x73_0 x73_1 -> (isum1 16 (\x74_0 -> (wvoc[x73_1][x74_0] F.* x70[x73_0][x74_0])))))
let x75 = (imap1 16 (\x76_0 -> (one F./ fromi64 16)))
let x77 = (let x78 = (imap2 16 27 (\x84_0 x84_1 -> ((F.neg x75[x84_0]) F.* target[x84_0][x84_1])))
in (let x79 = (imap2 16 27 (\x85_0 x85_1 -> (F.exp x72[x85_0][x85_1])))
in (let x80 = (imap1 16 (\x86_0 -> (one F./ (isum1 27 (\x87_0 -> x79[x86_0][x87_0])))))
in (let x81 = (imap1 16 (\x88_0 -> (isum1 27 (\x89_0 -> (let x90 = (imap1 27 (\x92_0 -> (F.exp x72[x88_0][x92_0])))
in (let x91 = (one F./ (isum1 27 (\x93_0 -> x90[x93_0])))
in ((x78[x88_0][x89_0] F.* (one F./ (x90[x89_0] F.* x91))) F.* x79[x88_0][x89_0])))))))
in (let x82 = (imap2 16 27 (\x94_0 x94_1 -> (let x95 = (imap1 27 (\x99_0 -> (F.exp x72[x94_0][x99_0])))
in (let x96 = (one F./ (isum1 27 (\x100_0 -> x95[x100_0])))
in (((x78[x94_0][x94_1] F.* (one F./ (x95[x94_1] F.* x96))) F.* x80[x94_0]) F.+ (F.neg (x81[x94_0] F.* (one F./ ((isum1 27 (\x97_0 -> x79[x94_0][x97_0])) F.* (isum1 27 (\x98_0 -> x79[x94_0][x98_0])))))))))))
in (imap2 16 27 (\x83_0 x83_1 -> ((F.exp x72[x83_0][x83_1]) F.* x82[x83_0][x83_1]))))))))
let x101 = (imap2 16 16 (\x102_0 x102_1 -> (isum1 27 (\x103_0 -> (x77[x102_0][x103_0] F.* wvoc[x103_0][x102_1])))))
let x104 = (imap2 16 64 (\x105_0 x105_1 -> (isum1 16 (\x106_0 -> (x101[x105_0][x106_0] F.* wdown[x106_0][x105_1])))))
let x107 = (imap2 16 64 (\x108_0 x108_1 -> ((indicatorp x62[x108_0][x108_1]) F.* x104[x108_0][x108_1])))
let x109 = (imap2 16 16 (\x110_0 x110_1 -> (isum1 64 (\x111_0 -> (x107[x110_0][x111_0] F.* wup[x111_0][x110_1])))))
let x112 = (let x113 = (imap2 16 16 (\x120_0 x120_1 -> (x53[x120_0][x120_1] F.* x53[x120_0][x120_1])))
in (let x114 = (imap1 16 (\x121_0 -> ((isum1 16 (\x122_0 -> x113[x121_0][x122_0])) F./ fromi64 16)))
in (let x115 = (imap1 16 (\x123_0 -> (F.sqrt (x114[x123_0] F.+ (one F./ fromi64 100000)))))
in (let x116 = (imap1 16 (\x124_0 -> (F.neg ((isum1 16 (\x125_0 -> (x109[x124_0][x125_0] F.* x53[x124_0][x125_0]))) F.* (one F./ (x115[x124_0] F.* x115[x124_0]))))))
in (let x117 = (imap1 16 (\x126_0 -> (x116[x126_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x114[x126_0] F.+ (one F./ fromi64 100000))))))))
in (let x118 = (imap2 16 16 (\x127_0 x127_1 -> (x117[x127_0] F./ fromi64 16)))
in (imap2 16 16 (\x119_0 x119_1 -> (x101[x119_0][x119_1] F.+ (((x109[x119_0][x119_1] F.* (one F./ x115[x119_0])) F.+ (x118[x119_0][x119_1] F.* x53[x119_0][x119_1])) F.+ (x118[x119_0][x119_1] F.* x53[x119_0][x119_1])))))))))))
let x128 = (imap2 16 16 (\x129_0 x129_1 -> (isum1 16 (\x130_0 -> (x112[x129_0][x130_0] F.* wout[x130_0][x129_1])))))
let x131 = (imap3 4 16 4 (\x132_0 x132_1 x132_2 -> x128[x132_1][((x132_0 * 4) + x132_2)]))
let x133 = (let x134 = (imap3 4 16 16 (\x139_0 x139_1 x139_2 -> (isum1 4 (\x140_0 -> (x25[x139_0][x139_1][x140_0] F.* x27[x139_0][x139_2][x140_0])))))
in (let x135 = (imap3 4 16 16 (\x141_0 x141_1 x141_2 -> ((x134[x141_0][x141_1][x141_2] F./ fromi64 2) F.+ mask[x141_1][x141_2])))
in (let x136 = (imap3 4 16 16 (\x142_0 x142_1 x142_2 -> (imaximum1 16 (\x143_0 -> x135[x142_0][x142_1][x143_0]))))
in (let x137 = (imap3 4 16 16 (\x144_0 x144_1 x144_2 -> (let x145 = (imap1 16 (\x147_0 -> (F.exp (x135[x144_0][x144_1][x147_0] F.+ (F.neg x136[x144_0][x144_1][x147_0])))))
in (let x146 = (one F./ (isum1 16 (\x148_0 -> x145[x148_0])))
in (x145[x144_2] F.* x146)))))
in (imap3 4 16 4 (\x138_0 x138_1 x138_2 -> (isum1 16 (\x149_0 -> (x131[x138_0][x149_0][x138_2] F.* x137[x138_0][x149_0][x138_1])))))))))
let x150 = (let x151 = (imap3 4 16 16 (\x168_0 x168_1 x168_2 -> (isum1 4 (\x169_0 -> (x25[x168_0][x168_1][x169_0] F.* x27[x168_0][x168_2][x169_0])))))
in (let x152 = (imap3 4 16 16 (\x170_0 x170_1 x170_2 -> ((x151[x170_0][x170_1][x170_2] F./ fromi64 2) F.+ mask[x170_1][x170_2])))
in (let x153 = (imap3 4 16 16 (\x171_0 x171_1 x171_2 -> (imaximum1 16 (\x172_0 -> x152[x171_0][x171_1][x172_0]))))
in (let x154 = (imap3 4 16 16 (\x173_0 x173_1 x173_2 -> (isum1 4 (\x174_0 -> (x131[x173_0][x173_1][x174_0] F.* x29[x173_0][x173_2][x174_0])))))
in (let x155 = (imap3 4 16 16 (\x175_0 x175_1 x175_2 -> (F.exp (x152[x175_0][x175_1][x175_2] F.+ (F.neg x153[x175_0][x175_1][x175_2])))))
in (let x156 = (imap2 4 16 (\x176_0 x176_1 -> (one F./ (isum1 16 (\x177_0 -> x155[x176_0][x176_1][x177_0])))))
in (let x157 = (imap2 4 16 (\x178_0 x178_1 -> (isum1 16 (\x179_0 -> (x154[x178_0][x178_1][x179_0] F.* x155[x178_0][x178_1][x179_0])))))
in (let x158 = (imap3 4 16 16 (\x180_0 x180_1 x180_2 -> ((x154[x180_0][x180_1][x180_2] F.* x156[x180_0][x180_1]) F.+ (F.neg (x157[x180_0][x180_1] F.* (one F./ ((isum1 16 (\x181_0 -> x155[x180_0][x180_1][x181_0])) F.* (isum1 16 (\x182_0 -> x155[x180_0][x180_1][x182_0])))))))))
in (let x159 = (imap3 4 16 16 (\x183_0 x183_1 x183_2 -> (F.neg ((F.exp (x152[x183_0][x183_1][x183_2] F.+ (F.neg x153[x183_0][x183_1][x183_2]))) F.* x158[x183_0][x183_1][x183_2]))))
in (let x160 = (imap2 4 16 (\x184_0 x184_1 -> x156[x184_0][x184_1]))
in (let x161 = (imap2 4 16 (\x185_0 x185_1 -> x157[x185_0][x185_1]))
in (let x162 = (imap3 4 16 16 (\x186_0 x186_1 x186_2 -> ((x154[x186_0][x186_1][x186_2] F.* x160[x186_0][x186_1]) F.+ (F.neg (x161[x186_0][x186_1] F.* (one F./ ((isum1 16 (\x187_0 -> x155[x186_0][x186_1][x187_0])) F.* (isum1 16 (\x188_0 -> x155[x186_0][x186_1][x188_0])))))))))
in (let x163 = (imap2 4 16 (\x189_0 x189_1 -> (imaximum1 16 (\x190_0 -> x152[x189_0][x189_1][x190_0]))))
in (let x164 = (imap2 4 16 (\x191_0 x191_1 -> (one F./ (isum1 16 (\x192_0 -> (one F.+ (F.neg (indicatorp (F.neg (x152[x191_0][x191_1][x192_0] F.+ (F.neg x163[x191_0][x191_1])))))))))))
in (let x165 = (imap3 4 16 16 (\x193_0 x193_1 x193_2 -> (((F.exp (x152[x193_0][x193_1][x193_2] F.+ (F.neg x153[x193_0][x193_1][x193_2]))) F.* x162[x193_0][x193_1][x193_2]) F.+ (((isum1 16 (\x194_0 -> x159[x193_0][x193_1][x194_0])) F.* (one F.+ (F.neg (indicatorp (F.neg (x152[x193_0][x193_1][x193_2] F.+ (F.neg x163[x193_0][x193_1]))))))) F.* x164[x193_0][x193_1]))))
in (let x166 = (imap3 4 16 16 (\x195_0 x195_1 x195_2 -> (x165[x195_0][x195_1][x195_2] F./ fromi64 2)))
in (imap3 4 16 4 (\x167_0 x167_1 x167_2 -> (isum1 16 (\x196_0 -> (x166[x167_0][x196_0][x167_1] F.* x25[x167_0][x196_0][x167_2])))))))))))))))))))))
let x197 = (let x198 = (imap3 4 16 16 (\x215_0 x215_1 x215_2 -> (isum1 4 (\x216_0 -> (x25[x215_0][x215_1][x216_0] F.* x27[x215_0][x215_2][x216_0])))))
in (let x199 = (imap3 4 16 16 (\x217_0 x217_1 x217_2 -> ((x198[x217_0][x217_1][x217_2] F./ fromi64 2) F.+ mask[x217_1][x217_2])))
in (let x200 = (imap3 4 16 16 (\x218_0 x218_1 x218_2 -> (imaximum1 16 (\x219_0 -> x199[x218_0][x218_1][x219_0]))))
in (let x201 = (imap3 4 16 16 (\x220_0 x220_1 x220_2 -> (isum1 4 (\x221_0 -> (x131[x220_0][x220_1][x221_0] F.* x29[x220_0][x220_2][x221_0])))))
in (let x202 = (imap3 4 16 16 (\x222_0 x222_1 x222_2 -> (F.exp (x199[x222_0][x222_1][x222_2] F.+ (F.neg x200[x222_0][x222_1][x222_2])))))
in (let x203 = (imap2 4 16 (\x223_0 x223_1 -> (one F./ (isum1 16 (\x224_0 -> x202[x223_0][x223_1][x224_0])))))
in (let x204 = (imap2 4 16 (\x225_0 x225_1 -> (isum1 16 (\x226_0 -> (x201[x225_0][x225_1][x226_0] F.* x202[x225_0][x225_1][x226_0])))))
in (let x205 = (imap3 4 16 16 (\x227_0 x227_1 x227_2 -> ((x201[x227_0][x227_1][x227_2] F.* x203[x227_0][x227_1]) F.+ (F.neg (x204[x227_0][x227_1] F.* (one F./ ((isum1 16 (\x228_0 -> x202[x227_0][x227_1][x228_0])) F.* (isum1 16 (\x229_0 -> x202[x227_0][x227_1][x229_0])))))))))
in (let x206 = (imap3 4 16 16 (\x230_0 x230_1 x230_2 -> (F.neg ((F.exp (x199[x230_0][x230_1][x230_2] F.+ (F.neg x200[x230_0][x230_1][x230_2]))) F.* x205[x230_0][x230_1][x230_2]))))
in (let x207 = (imap2 4 16 (\x231_0 x231_1 -> x203[x231_0][x231_1]))
in (let x208 = (imap2 4 16 (\x232_0 x232_1 -> x204[x232_0][x232_1]))
in (let x209 = (imap3 4 16 16 (\x233_0 x233_1 x233_2 -> ((x201[x233_0][x233_1][x233_2] F.* x207[x233_0][x233_1]) F.+ (F.neg (x208[x233_0][x233_1] F.* (one F./ ((isum1 16 (\x234_0 -> x202[x233_0][x233_1][x234_0])) F.* (isum1 16 (\x235_0 -> x202[x233_0][x233_1][x235_0])))))))))
in (let x210 = (imap2 4 16 (\x236_0 x236_1 -> (imaximum1 16 (\x237_0 -> x199[x236_0][x236_1][x237_0]))))
in (let x211 = (imap2 4 16 (\x238_0 x238_1 -> (one F./ (isum1 16 (\x239_0 -> (one F.+ (F.neg (indicatorp (F.neg (x199[x238_0][x238_1][x239_0] F.+ (F.neg x210[x238_0][x238_1])))))))))))
in (let x212 = (imap3 4 16 16 (\x240_0 x240_1 x240_2 -> (((F.exp (x199[x240_0][x240_1][x240_2] F.+ (F.neg x200[x240_0][x240_1][x240_2]))) F.* x209[x240_0][x240_1][x240_2]) F.+ (((isum1 16 (\x241_0 -> x206[x240_0][x240_1][x241_0])) F.* (one F.+ (F.neg (indicatorp (F.neg (x199[x240_0][x240_1][x240_2] F.+ (F.neg x210[x240_0][x240_1]))))))) F.* x211[x240_0][x240_1]))))
in (let x213 = (imap3 4 16 16 (\x242_0 x242_1 x242_2 -> (x212[x242_0][x242_1][x242_2] F./ fromi64 2)))
in (imap3 4 16 4 (\x214_0 x214_1 x214_2 -> (isum1 16 (\x243_0 -> (x213[x214_0][x214_1][x243_0] F.* x27[x214_0][x243_0][x214_2])))))))))))))))))))))
let x244 = (imap2 16 16 (\x245_0 x245_1 -> x133[(x245_1 / 4)][x245_0][(x245_1 % 4)]))
let x246 = (imap2 16 16 (\x247_0 x247_1 -> x150[(x247_1 / 4)][x247_0][(x247_1 % 4)]))
let x248 = (imap2 16 16 (\x249_0 x249_1 -> x197[(x249_1 / 4)][x249_0][(x249_1 % 4)]))
let x250 = (imap2 16 16 (\x251_0 x251_1 -> (((isum1 16 (\x252_0 -> (x244[x251_0][x252_0] F.* wval[x252_0][x251_1]))) F.+ (isum1 16 (\x253_0 -> (x246[x251_0][x253_0] F.* wkey[x253_0][x251_1])))) F.+ (isum1 16 (\x254_0 -> (x248[x251_0][x254_0] F.* wqry[x254_0][x251_1]))))))
let x255 = (let x256 = (imap2 16 16 (\x263_0 x263_1 -> (x2[x263_0][x263_1] F.* x2[x263_0][x263_1])))
in (let x257 = (imap1 16 (\x264_0 -> ((isum1 16 (\x265_0 -> x256[x264_0][x265_0])) F./ fromi64 16)))
in (let x258 = (imap1 16 (\x266_0 -> (F.sqrt (x257[x266_0] F.+ (one F./ fromi64 100000)))))
in (let x259 = (imap1 16 (\x267_0 -> (F.neg ((isum1 16 (\x268_0 -> (x250[x267_0][x268_0] F.* x2[x267_0][x268_0]))) F.* (one F./ (x258[x267_0] F.* x258[x267_0]))))))
in (let x260 = (imap1 16 (\x269_0 -> (x259[x269_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x257[x269_0] F.+ (one F./ fromi64 100000))))))))
in (let x261 = (imap2 16 16 (\x270_0 x270_1 -> (x260[x270_0] F./ fromi64 16)))
in (imap2 16 16 (\x262_0 x262_1 -> (x112[x262_0][x262_1] F.+ (((x250[x262_0][x262_1] F.* (one F./ x258[x262_0])) F.+ (x261[x262_0][x262_1] F.* x2[x262_0][x262_1])) F.+ (x261[x262_0][x262_1] F.* x2[x262_0][x262_1])))))))))))
let x271 = (let x272 = (imap2 16 16 (\x279_0 x279_1 -> (x0[x279_0][x279_1] F.* x0[x279_0][x279_1])))
in (let x273 = (imap1 16 (\x280_0 -> ((isum1 16 (\x281_0 -> x272[x280_0][x281_0])) F./ fromi64 16)))
in (let x274 = (imap1 16 (\x282_0 -> (F.sqrt (x273[x282_0] F.+ (one F./ fromi64 100000)))))
in (let x275 = (imap1 16 (\x283_0 -> (F.neg ((isum1 16 (\x284_0 -> (x255[x283_0][x284_0] F.* x0[x283_0][x284_0]))) F.* (one F./ (x274[x283_0] F.* x274[x283_0]))))))
in (let x276 = (imap1 16 (\x285_0 -> (x275[x285_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x273[x285_0] F.+ (one F./ fromi64 100000))))))))
in (let x277 = (imap2 16 16 (\x286_0 x286_1 -> (x276[x286_0] F./ fromi64 16)))
in (imap2 16 16 (\x278_0 x278_1 -> (((x255[x278_0][x278_1] F.* (one F./ x274[x278_0])) F.+ (x277[x278_0][x278_1] F.* x0[x278_0][x278_1])) F.+ (x277[x278_0][x278_1] F.* x0[x278_0][x278_1]))))))))))

let dmask = (let x287 = (imap3 4 16 16 (\x304_0 x304_1 x304_2 -> (isum1 4 (\x305_0 -> (x25[x304_0][x304_1][x305_0] F.* x27[x304_0][x304_2][x305_0])))))
in (let x288 = (imap3 4 16 16 (\x306_0 x306_1 x306_2 -> ((x287[x306_0][x306_1][x306_2] F./ fromi64 2) F.+ mask[x306_1][x306_2])))
in (let x289 = (imap3 4 16 16 (\x307_0 x307_1 x307_2 -> (imaximum1 16 (\x308_0 -> x288[x307_0][x307_1][x308_0]))))
in (let x290 = (imap3 4 16 16 (\x309_0 x309_1 x309_2 -> (isum1 4 (\x310_0 -> (x131[x309_0][x309_1][x310_0] F.* x29[x309_0][x309_2][x310_0])))))
in (let x291 = (imap3 4 16 16 (\x311_0 x311_1 x311_2 -> (F.exp (x288[x311_0][x311_1][x311_2] F.+ (F.neg x289[x311_0][x311_1][x311_2])))))
in (let x292 = (imap2 4 16 (\x312_0 x312_1 -> (one F./ (isum1 16 (\x313_0 -> x291[x312_0][x312_1][x313_0])))))
in (let x293 = (imap2 4 16 (\x314_0 x314_1 -> (isum1 16 (\x315_0 -> (x290[x314_0][x314_1][x315_0] F.* x291[x314_0][x314_1][x315_0])))))
in (let x294 = (imap3 4 16 16 (\x316_0 x316_1 x316_2 -> ((x290[x316_0][x316_1][x316_2] F.* x292[x316_0][x316_1]) F.+ (F.neg (x293[x316_0][x316_1] F.* (one F./ ((isum1 16 (\x317_0 -> x291[x316_0][x316_1][x317_0])) F.* (isum1 16 (\x318_0 -> x291[x316_0][x316_1][x318_0])))))))))
in (let x295 = (imap3 4 16 16 (\x319_0 x319_1 x319_2 -> (F.neg ((F.exp (x288[x319_0][x319_1][x319_2] F.+ (F.neg x289[x319_0][x319_1][x319_2]))) F.* x294[x319_0][x319_1][x319_2]))))
in (let x296 = (imap2 4 16 (\x320_0 x320_1 -> x292[x320_0][x320_1]))
in (let x297 = (imap2 4 16 (\x321_0 x321_1 -> x293[x321_0][x321_1]))
in (let x298 = (imap3 4 16 16 (\x322_0 x322_1 x322_2 -> ((x290[x322_0][x322_1][x322_2] F.* x296[x322_0][x322_1]) F.+ (F.neg (x297[x322_0][x322_1] F.* (one F./ ((isum1 16 (\x323_0 -> x291[x322_0][x322_1][x323_0])) F.* (isum1 16 (\x324_0 -> x291[x322_0][x322_1][x324_0])))))))))
in (let x299 = (imap2 4 16 (\x325_0 x325_1 -> (imaximum1 16 (\x326_0 -> x288[x325_0][x325_1][x326_0]))))
in (let x300 = (imap2 4 16 (\x327_0 x327_1 -> (one F./ (isum1 16 (\x328_0 -> (one F.+ (F.neg (indicatorp (F.neg (x288[x327_0][x327_1][x328_0] F.+ (F.neg x299[x327_0][x327_1])))))))))))
in (let x301 = (imap3 4 16 16 (\x329_0 x329_1 x329_2 -> (((F.exp (x288[x329_0][x329_1][x329_2] F.+ (F.neg x289[x329_0][x329_1][x329_2]))) F.* x298[x329_0][x329_1][x329_2]) F.+ (((isum1 16 (\x330_0 -> x295[x329_0][x329_1][x330_0])) F.* (one F.+ (F.neg (indicatorp (F.neg (x288[x329_0][x329_1][x329_2] F.+ (F.neg x299[x329_0][x329_1]))))))) F.* x300[x329_0][x329_1]))))
in (imap2 16 16 (\x303_0 x303_1 -> (isum1 4 (\x302_0 -> x301[x302_0][x303_0][x303_1])))))))))))))))))))
let dwpe = (imap2 16 16 (\x331_0 x331_1 -> x271[x331_0][x331_1]))
let dwqry = (imap2 16 16 (\x332_0 x332_1 -> (isum1 16 (\x333_0 -> (x248[x333_0][x332_0] F.* x9[x333_0][x332_1])))))
let dwkey = (imap2 16 16 (\x334_0 x334_1 -> (isum1 16 (\x335_0 -> (x246[x335_0][x334_0] F.* x9[x335_0][x334_1])))))
let dwval = (imap2 16 16 (\x336_0 x336_1 -> (isum1 16 (\x337_0 -> (x244[x337_0][x336_0] F.* x9[x337_0][x336_1])))))
let dwout = (imap2 16 16 (\x338_0 x338_1 -> (isum1 16 (\x339_0 -> (x112[x339_0][x338_0] F.* x48[x339_0][x338_1])))))
let dwup = (imap2 64 16 (\x340_0 x340_1 -> (isum1 16 (\x341_0 -> (x107[x341_0][x340_0] F.* x55[x341_0][x340_1])))))
let dwdown = (imap2 16 64 (\x342_0 x342_1 -> (isum1 16 (\x343_0 -> (x101[x343_0][x342_0] F.* x65[x343_0][x342_1])))))
let dwvoc = (imap2 27 16 (\x344_0 x344_1 -> (isum1 16 (\x345_0 -> (x77[x345_0][x344_0] F.* x70[x345_0][x344_1])))))
let dwseq = (imap2 16 16 (\x346_0 x346_1 -> x271[x346_0][x346_1]))
let dtarget = (let x347 = (imap2 16 27 (\x349_0 x349_1 -> (let x350 = (imap1 27 (\x352_0 -> (F.exp x72[x349_0][x352_0])))
in (let x351 = (one F./ (isum1 27 (\x353_0 -> x350[x353_0])))
in (F.log (x350[x349_1] F.* x351))))))
in (imap2 16 27 (\x348_0 x348_1 -> ((F.neg x75[x348_0]) F.* x347[x348_0][x348_1]))))

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
  let lt_r = 0.01 * (1 - (nn64.fromi64 step) / (nn64.fromi64 0050))
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
  (masks : [0050][16][16]f64) (dls : [0050]i64)
  (seqs : [0050][16]i64) =
  let (new_p, new_mp, new_vp) =
    loop (p', mp', vp') = (p, mp, vp)
    for step < 0050 do
      let dl = dls[step]
      let tokens = seqs[step]
      let mask = masks[step]
      in (cal_step dl p' mp' vp' tokens mask step)
  in ((from_params new_p), (from_params new_mp), (from_params new_vp))

entry test (inp : [2]f64) =
  let x0 = ( nn64.isum1 2 (\x1_0 -> inp[x1_0]))
  let dinp = (imap1 2 (\x2_0 -> (x0 + x0)))
  in dinp

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