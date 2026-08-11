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
let x31 = (imap3 4 16 4 (\x32_0 x32_1 x32_2 -> (let x33 = (imap2 16 16 (\x36_0 x36_1 -> (isum1 4 (\x37_0 -> (x25[x32_0][x36_0][x37_0] F.* x27[x32_0][x36_1][x37_0])))))
in (let x34 = (imap2 16 16 (\x38_0 x38_1 -> ((x33[x38_0][x38_1] F./ fromi64 2) F.+ mask[x38_0][x38_1])))
in (let x35 = (imap2 16 16 (\x39_0 x39_1 -> (let x40 = (imaximum1 16 (\x43_0 -> x34[x39_0][x43_0]))
in (let x41 = (imap1 16 (\x44_0 -> (F.exp (x34[x39_0][x44_0] F.+ (F.neg x40)))))
in (let x42 = (isum1 16 (\x45_0 -> x41[x45_0]))
in (x41[x39_1] F.* (one F./ x42)))))))
in (isum1 16 (\x46_0 -> (x35[x32_1][x46_0] F.* x29[x32_0][x46_0][x32_2]))))))))
let x47 = (imap2 16 16 (\x48_0 x48_1 -> x31[(x48_1 / 4)][x48_0][(x48_1 % 4)]))
let x49 = (imap2 16 16 (\x50_0 x50_1 -> (isum1 16 (\x51_0 -> (wout[x50_1][x51_0] F.* x47[x50_0][x51_0])))))
let x52 = (imap2 16 16 (\x53_0 x53_1 -> (x49[x53_0][x53_1] F.+ x2[x53_0][x53_1])))
let x54 = (imap2 16 16 (\x55_0 x55_1 -> (let x56 = (imap1 16 (\x59_0 -> (x52[x55_0][x59_0] F.* x52[x55_0][x59_0])))
in (let x57 = ((isum1 16 (\x60_0 -> x56[x60_0])) F./ fromi64 16)
in (let x58 = (F.sqrt (x57 F.+ (one F./ fromi64 100000)))
in (x52[x55_0][x55_1] F.* (one F./ x58)))))))
let x61 = (imap2 16 64 (\x62_0 x62_1 -> (isum1 16 (\x63_0 -> (wup[x62_1][x63_0] F.* x54[x62_0][x63_0])))))
let x64 = (imap2 16 64 (\x65_0 x65_1 -> F.max x61[x65_0][x65_1] zero))
let x66 = (imap2 16 16 (\x67_0 x67_1 -> (isum1 64 (\x68_0 -> (wdown[x67_1][x68_0] F.* x64[x67_0][x68_0])))))
let x69 = (imap2 16 16 (\x70_0 x70_1 -> (x66[x70_0][x70_1] F.+ x52[x70_0][x70_1])))
let x71 = (imap2 16 27 (\x72_0 x72_1 -> (isum1 16 (\x73_0 -> (wvoc[x72_1][x73_0] F.* x69[x72_0][x73_0])))))
let x74 = (imap1 16 (\x75_0 -> (one F./ fromi64 16)))
let x76 = (imap2 16 27 (\x77_0 x77_1 -> (let x78 = (imaximum1 27 (\x81_0 -> x71[x77_0][x81_0]))
in (let x79 = (imap1 27 (\x82_0 -> (F.exp (x71[x77_0][x82_0] F.+ (F.neg x78)))))
in (let x80 = (isum1 27 (\x83_0 -> x79[x83_0]))
in (F.log (x79[x77_1] F.* (one F./ x80))))))))
let x84 = (imap2 16 27 (\x85_0 x85_1 -> ((F.neg x74[x85_0]) F.* target[x85_0][x85_1])))
let x86 = (imap1 16 (\x87_0 -> (imaximum1 27 (\x88_0 -> x71[x87_0][x88_0]))))
let x89 = (imap2 16 27 (\x90_0 x90_1 -> (F.exp (x71[x90_0][x90_1] F.+ (F.neg x86[x90_0])))))
let x91 = (imap1 16 (\x92_0 -> (isum1 27 (\x93_0 -> x89[x92_0][x93_0]))))
let x94 = (imap1 16 (\x95_0 -> (isum1 27 (\x96_0 -> (let x97 = (imaximum1 27 (\x100_0 -> x71[x95_0][x100_0]))
in (let x98 = (imap1 27 (\x101_0 -> (F.exp (x71[x95_0][x101_0] F.+ (F.neg x97)))))
in (let x99 = (isum1 27 (\x102_0 -> x98[x102_0]))
in (F.neg (((x84[x95_0][x96_0] F.* (one F./ (x98[x96_0] F.* (one F./ x99)))) F.* x89[x95_0][x96_0]) F.* (one F./ (x91[x95_0] F.* x91[x95_0])))))))))))
let x103 = (imap2 16 27 (\x104_0 x104_1 -> (let x105 = (imaximum1 27 (\x108_0 -> x71[x104_0][x108_0]))
in (let x106 = (imap1 27 (\x109_0 -> (F.exp (x71[x104_0][x109_0] F.+ (F.neg x105)))))
in (let x107 = (isum1 27 (\x110_0 -> x106[x110_0]))
in (((x84[x104_0][x104_1] F.* (one F./ (x106[x104_1] F.* (one F./ x107)))) F.* (one F./ x91[x104_0])) F.+ x94[x104_0]))))))
let x111 = (imap1 16 (\x112_0 -> (isum1 27 (\x113_0 -> (F.neg ((F.exp (x71[x112_0][x113_0] F.+ (F.neg x86[x112_0]))) F.* x103[x112_0][x113_0]))))))
let x114 = (imap1 16 (\x115_0 -> (one F./ (isum1 27 (\x116_0 -> (one F.+ (F.neg (indicatorp (F.neg (x71[x115_0][x116_0] F.+ (F.neg x86[x115_0])))))))))))
let x117 = (imap2 16 27 (\x118_0 x118_1 -> (((F.exp (x71[x118_0][x118_1] F.+ (F.neg x86[x118_0]))) F.* x103[x118_0][x118_1]) F.+ ((x111[x118_0] F.* (one F.+ (F.neg (indicatorp (F.neg (x71[x118_0][x118_1] F.+ (F.neg x86[x118_0]))))))) F.* x114[x118_0]))))
let x119 = (imap2 16 16 (\x120_0 x120_1 -> (isum1 27 (\x121_0 -> (x117[x120_0][x121_0] F.* wvoc[x121_0][x120_1])))))
let x122 = (imap2 16 64 (\x123_0 x123_1 -> (isum1 16 (\x124_0 -> (x119[x123_0][x124_0] F.* wdown[x124_0][x123_1])))))
let x125 = (imap2 16 64 (\x126_0 x126_1 -> ((indicatorp x61[x126_0][x126_1]) F.* x122[x126_0][x126_1])))
let x127 = (imap2 16 16 (\x128_0 x128_1 -> (isum1 64 (\x129_0 -> (x125[x128_0][x129_0] F.* wup[x129_0][x128_1])))))
let x130 = (imap2 16 16 (\x131_0 x131_1 -> (x52[x131_0][x131_1] F.* x52[x131_0][x131_1])))
let x132 = (imap1 16 (\x133_0 -> ((isum1 16 (\x134_0 -> x130[x133_0][x134_0])) F./ fromi64 16)))
let x135 = (imap1 16 (\x136_0 -> (F.sqrt (x132[x136_0] F.+ (one F./ fromi64 100000)))))
let x137 = (imap1 16 (\x138_0 -> (isum1 16 (\x139_0 -> (F.neg ((x127[x138_0][x139_0] F.* x52[x138_0][x139_0]) F.* (one F./ (x135[x138_0] F.* x135[x138_0]))))))))
let x140 = (imap1 16 (\x141_0 -> (x137[x141_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x132[x141_0] F.+ (one F./ fromi64 100000))))))))
let x142 = (imap2 16 16 (\x143_0 x143_1 -> (x140[x143_0] F./ fromi64 16)))
let x144 = (imap2 16 16 (\x145_0 x145_1 -> (x119[x145_0][x145_1] F.+ (((x127[x145_0][x145_1] F.* (one F./ x135[x145_0])) F.+ (x142[x145_0][x145_1] F.* x52[x145_0][x145_1])) F.+ (x142[x145_0][x145_1] F.* x52[x145_0][x145_1])))))
let x146 = (imap2 16 16 (\x147_0 x147_1 -> (isum1 16 (\x148_0 -> (x144[x147_0][x148_0] F.* wout[x148_0][x147_1])))))
let x149 = (imap3 4 16 4 (\x150_0 x150_1 x150_2 -> x146[x150_1][((x150_0 * 4) + x150_2)]))
let x151 = (imap3 4 16 16 (\x152_0 x152_1 x152_2 -> (isum1 4 (\x153_0 -> (x25[x152_0][x152_1][x153_0] F.* x27[x152_0][x152_2][x153_0])))))
let x154 = (imap3 4 16 16 (\x155_0 x155_1 x155_2 -> ((x151[x155_0][x155_1][x155_2] F./ fromi64 2) F.+ mask[x155_1][x155_2])))
let x156 = (imap3 4 16 16 (\x157_0 x157_1 x157_2 -> (let x158 = (imaximum1 16 (\x161_0 -> x154[x157_0][x157_1][x161_0]))
in (let x159 = (imap1 16 (\x162_0 -> (F.exp (x154[x157_0][x157_1][x162_0] F.+ (F.neg x158)))))
in (let x160 = (isum1 16 (\x163_0 -> x159[x163_0]))
in (x159[x157_2] F.* (one F./ x160)))))))
let x164 = (imap3 4 16 16 (\x165_0 x165_1 x165_2 -> (isum1 4 (\x166_0 -> (x149[x165_0][x165_1][x166_0] F.* x29[x165_0][x165_2][x166_0])))))
let x167 = (imap2 4 16 (\x168_0 x168_1 -> (imaximum1 16 (\x169_0 -> x154[x168_0][x168_1][x169_0]))))
let x170 = (imap3 4 16 16 (\x171_0 x171_1 x171_2 -> (F.exp (x154[x171_0][x171_1][x171_2] F.+ (F.neg x167[x171_0][x171_1])))))
let x172 = (imap2 4 16 (\x173_0 x173_1 -> (isum1 16 (\x174_0 -> x170[x173_0][x173_1][x174_0]))))
let x175 = (imap2 4 16 (\x176_0 x176_1 -> (isum1 16 (\x177_0 -> (F.neg ((x164[x176_0][x176_1][x177_0] F.* x170[x176_0][x176_1][x177_0]) F.* (one F./ (x172[x176_0][x176_1] F.* x172[x176_0][x176_1]))))))))
let x178 = (imap3 4 16 16 (\x179_0 x179_1 x179_2 -> ((x164[x179_0][x179_1][x179_2] F.* (one F./ x172[x179_0][x179_1])) F.+ x175[x179_0][x179_1])))
let x180 = (imap2 4 16 (\x181_0 x181_1 -> (isum1 16 (\x182_0 -> (F.neg ((F.exp (x154[x181_0][x181_1][x182_0] F.+ (F.neg x167[x181_0][x181_1]))) F.* x178[x181_0][x181_1][x182_0]))))))
let x183 = (imap2 4 16 (\x184_0 x184_1 -> (one F./ (isum1 16 (\x185_0 -> (one F.+ (F.neg (indicatorp (F.neg (x154[x184_0][x184_1][x185_0] F.+ (F.neg x167[x184_0][x184_1])))))))))))
let x186 = (imap3 4 16 16 (\x187_0 x187_1 x187_2 -> (((F.exp (x154[x187_0][x187_1][x187_2] F.+ (F.neg x167[x187_0][x187_1]))) F.* x178[x187_0][x187_1][x187_2]) F.+ ((x180[x187_0][x187_1] F.* (one F.+ (F.neg (indicatorp (F.neg (x154[x187_0][x187_1][x187_2] F.+ (F.neg x167[x187_0][x187_1]))))))) F.* x183[x187_0][x187_1]))))
let x188 = (imap3 4 16 16 (\x189_0 x189_1 x189_2 -> (x186[x189_0][x189_1][x189_2] F./ fromi64 2)))
let x190 = (imap3 4 16 4 (\x191_0 x191_1 x191_2 -> (isum1 16 (\x192_0 -> (x149[x191_0][x192_0][x191_2] F.* x156[x191_0][x192_0][x191_1])))))
let x193 = (imap3 4 16 4 (\x194_0 x194_1 x194_2 -> (isum1 16 (\x195_0 -> (x188[x194_0][x195_0][x194_1] F.* x25[x194_0][x195_0][x194_2])))))
let x196 = (imap3 4 16 4 (\x197_0 x197_1 x197_2 -> (isum1 16 (\x198_0 -> (x188[x197_0][x197_1][x198_0] F.* x27[x197_0][x198_0][x197_2])))))
let x199 = (imap2 16 16 (\x200_0 x200_1 -> x190[(x200_1 / 4)][x200_0][(x200_1 % 4)]))
let x201 = (imap2 16 16 (\x202_0 x202_1 -> x193[(x202_1 / 4)][x202_0][(x202_1 % 4)]))
let x203 = (imap2 16 16 (\x204_0 x204_1 -> x196[(x204_1 / 4)][x204_0][(x204_1 % 4)]))
let x205 = (imap2 16 16 (\x206_0 x206_1 -> (((isum1 16 (\x207_0 -> (x199[x206_0][x207_0] F.* wval[x207_0][x206_1]))) F.+ (isum1 16 (\x208_0 -> (x201[x206_0][x208_0] F.* wkey[x208_0][x206_1])))) F.+ (isum1 16 (\x209_0 -> (x203[x206_0][x209_0] F.* wqry[x209_0][x206_1]))))))
let x210 = (imap2 16 16 (\x211_0 x211_1 -> (x2[x211_0][x211_1] F.* x2[x211_0][x211_1])))
let x212 = (imap1 16 (\x213_0 -> ((isum1 16 (\x214_0 -> x210[x213_0][x214_0])) F./ fromi64 16)))
let x215 = (imap1 16 (\x216_0 -> (F.sqrt (x212[x216_0] F.+ (one F./ fromi64 100000)))))
let x217 = (imap1 16 (\x218_0 -> (isum1 16 (\x219_0 -> (F.neg ((x205[x218_0][x219_0] F.* x2[x218_0][x219_0]) F.* (one F./ (x215[x218_0] F.* x215[x218_0]))))))))
let x220 = (imap1 16 (\x221_0 -> (x217[x221_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x212[x221_0] F.+ (one F./ fromi64 100000))))))))
let x222 = (imap2 16 16 (\x223_0 x223_1 -> (x220[x223_0] F./ fromi64 16)))
let x224 = (imap2 16 16 (\x225_0 x225_1 -> (x144[x225_0][x225_1] F.+ (((x205[x225_0][x225_1] F.* (one F./ x215[x225_0])) F.+ (x222[x225_0][x225_1] F.* x2[x225_0][x225_1])) F.+ (x222[x225_0][x225_1] F.* x2[x225_0][x225_1])))))
let x226 = (imap2 16 16 (\x227_0 x227_1 -> (x0[x227_0][x227_1] F.* x0[x227_0][x227_1])))
let x228 = (imap1 16 (\x229_0 -> ((isum1 16 (\x230_0 -> x226[x229_0][x230_0])) F./ fromi64 16)))
let x231 = (imap1 16 (\x232_0 -> (F.sqrt (x228[x232_0] F.+ (one F./ fromi64 100000)))))
let x233 = (imap1 16 (\x234_0 -> (isum1 16 (\x235_0 -> (F.neg ((x224[x234_0][x235_0] F.* x0[x234_0][x235_0]) F.* (one F./ (x231[x234_0] F.* x231[x234_0]))))))))
let x236 = (imap1 16 (\x237_0 -> (x233[x237_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x228[x237_0] F.+ (one F./ fromi64 100000))))))))
let x238 = (imap2 16 16 (\x239_0 x239_1 -> (x236[x239_0] F./ fromi64 16)))
let x240 = (imap2 16 16 (\x241_0 x241_1 -> (((x224[x241_0][x241_1] F.* (one F./ x231[x241_0])) F.+ (x238[x241_0][x241_1] F.* x0[x241_0][x241_1])) F.+ (x238[x241_0][x241_1] F.* x0[x241_0][x241_1]))))

let dmask = (imap2 16 16 (\x243_0 x243_1 -> (isum1 4 (\x242_0 -> x186[x242_0][x243_0][x243_1]))))
let dwpe = (imap2 16 16 (\x244_0 x244_1 -> x240[x244_0][x244_1]))
let dwqry = (imap2 16 16 (\x245_0 x245_1 -> (isum1 16 (\x246_0 -> (x203[x246_0][x245_0] F.* x9[x246_0][x245_1])))))
let dwkey = (imap2 16 16 (\x247_0 x247_1 -> (isum1 16 (\x248_0 -> (x201[x248_0][x247_0] F.* x9[x248_0][x247_1])))))
let dwval = (imap2 16 16 (\x249_0 x249_1 -> (isum1 16 (\x250_0 -> (x199[x250_0][x249_0] F.* x9[x250_0][x249_1])))))
let dwout = (imap2 16 16 (\x251_0 x251_1 -> (isum1 16 (\x252_0 -> (x144[x252_0][x251_0] F.* x47[x252_0][x251_1])))))
let dwup = (imap2 64 16 (\x253_0 x253_1 -> (isum1 16 (\x254_0 -> (x125[x254_0][x253_0] F.* x54[x254_0][x253_1])))))
let dwdown = (imap2 16 64 (\x255_0 x255_1 -> (isum1 16 (\x256_0 -> (x119[x256_0][x255_0] F.* x64[x256_0][x255_1])))))
let dwvoc = (imap2 27 16 (\x257_0 x257_1 -> (isum1 16 (\x258_0 -> (x117[x258_0][x257_0] F.* x69[x258_0][x257_1])))))
let dwseq = (imap2 16 16 (\x259_0 x259_1 -> x240[x259_0][x259_1]))
let dtarget = (imap2 16 27 (\x260_0 x260_1 -> ((F.neg x74[x260_0]) F.* x76[x260_0][x260_1])))

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