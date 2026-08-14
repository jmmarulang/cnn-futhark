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
let x31 = (imap3 4 16 4 (\x32_0 x32_1 x32_2 -> (let x33 = (imap2 16 16 (\x34_0 x34_1 -> (let x35 = (imaximum1 16 (\x38_0 -> (((isum1 4 (\x39_0 -> (x25[x32_0][x34_0][x39_0] F.* x27[x32_0][x38_0][x39_0]))) F./ fromi64 2) F.+ mask[x34_0][x38_0])))
in (let x36 = (imap1 16 (\x40_0 -> (F.exp ((((isum1 4 (\x41_0 -> (x25[x32_0][x34_0][x41_0] F.* x27[x32_0][x40_0][x41_0]))) F./ fromi64 2) F.+ mask[x34_0][x40_0]) F.+ (F.neg x35)))))
in (let x37 = (isum1 16 (\x42_0 -> x36[x42_0]))
in (x36[x34_1] F.* (one F./ x37)))))))
in (isum1 16 (\x43_0 -> (x33[x32_1][x43_0] F.* x29[x32_0][x43_0][x32_2]))))))
let x44 = (imap2 16 16 (\x45_0 x45_1 -> x31[(x45_1 / 4)][x45_0][(x45_1 % 4)]))
let x46 = (imap2 16 16 (\x47_0 x47_1 -> (isum1 16 (\x48_0 -> (wout[x47_1][x48_0] F.* x44[x47_0][x48_0])))))
let x49 = (imap2 16 16 (\x50_0 x50_1 -> (x46[x50_0][x50_1] F.+ x2[x50_0][x50_1])))
let x51 = (imap2 16 16 (\x52_0 x52_1 -> (let x53 = (imap1 16 (\x56_0 -> (x49[x52_0][x56_0] F.* x49[x52_0][x56_0])))
in (let x54 = ((isum1 16 (\x57_0 -> x53[x57_0])) F./ fromi64 16)
in (let x55 = (F.sqrt (x54 F.+ (one F./ fromi64 100000)))
in (x49[x52_0][x52_1] F.* (one F./ x55)))))))
let x58 = (imap2 16 64 (\x59_0 x59_1 -> (isum1 16 (\x60_0 -> (wup[x59_1][x60_0] F.* x51[x59_0][x60_0])))))
let x61 = (imap2 16 64 (\x62_0 x62_1 -> F.max x58[x62_0][x62_1] zero))
let x63 = (imap2 16 16 (\x64_0 x64_1 -> (isum1 64 (\x65_0 -> (wdown[x64_1][x65_0] F.* x61[x64_0][x65_0])))))
let x66 = (imap2 16 16 (\x67_0 x67_1 -> (x63[x67_0][x67_1] F.+ x49[x67_0][x67_1])))
let x68 = (imap2 16 27 (\x69_0 x69_1 -> (isum1 16 (\x70_0 -> (wvoc[x69_1][x70_0] F.* x66[x69_0][x70_0])))))
let x71 = (imap1 16 (\x72_0 -> (one F./ fromi64 16)))
let x73 = (let x74 = (imap2 16 27 (\x83_0 x83_1 -> ((F.neg x71[x83_0]) F.* target[x83_0][x83_1])))
in (let x75 = (imap1 16 (\x84_0 -> (imaximum1 27 (\x85_0 -> x68[x84_0][x85_0]))))
in (let x76 = (imap2 16 27 (\x86_0 x86_1 -> (F.exp (x68[x86_0][x86_1] F.+ (F.neg x75[x86_0])))))
in (let x77 = (imap1 16 (\x87_0 -> (isum1 27 (\x88_0 -> x76[x87_0][x88_0]))))
in (let x78 = (imap1 16 (\x89_0 -> (isum1 27 (\x90_0 -> (let x91 = (imaximum1 27 (\x94_0 -> x68[x89_0][x94_0]))
in (let x92 = (imap1 27 (\x95_0 -> (F.exp (x68[x89_0][x95_0] F.+ (F.neg x91)))))
in (let x93 = (isum1 27 (\x96_0 -> x92[x96_0]))
in (F.neg (((x74[x89_0][x90_0] F.* (one F./ (x92[x90_0] F.* (one F./ x93)))) F.* x76[x89_0][x90_0]) F.* (one F./ (x77[x89_0] F.* x77[x89_0])))))))))))
in (let x79 = (imap2 16 27 (\x97_0 x97_1 -> (let x98 = (imaximum1 27 (\x101_0 -> x68[x97_0][x101_0]))
in (let x99 = (imap1 27 (\x102_0 -> (F.exp (x68[x97_0][x102_0] F.+ (F.neg x98)))))
in (let x100 = (isum1 27 (\x103_0 -> x99[x103_0]))
in (((x74[x97_0][x97_1] F.* (one F./ (x99[x97_1] F.* (one F./ x100)))) F.* (one F./ x77[x97_0])) F.+ x78[x97_0]))))))
in (let x80 = (imap1 16 (\x104_0 -> (isum1 27 (\x105_0 -> (F.neg ((F.exp (x68[x104_0][x105_0] F.+ (F.neg x75[x104_0]))) F.* x79[x104_0][x105_0]))))))
in (let x81 = (imap1 16 (\x106_0 -> (one F./ (isum1 27 (\x107_0 -> (one F.+ (F.neg (indicatorp (F.neg (x68[x106_0][x107_0] F.+ (F.neg x75[x106_0])))))))))))
in (imap2 16 27 (\x82_0 x82_1 -> (((F.exp (x68[x82_0][x82_1] F.+ (F.neg x75[x82_0]))) F.* x79[x82_0][x82_1]) F.+ ((x80[x82_0] F.* (one F.+ (F.neg (indicatorp (F.neg (x68[x82_0][x82_1] F.+ (F.neg x75[x82_0]))))))) F.* x81[x82_0]))))))))))))
let x108 = (imap2 16 16 (\x109_0 x109_1 -> (isum1 27 (\x110_0 -> (x73[x109_0][x110_0] F.* wvoc[x110_0][x109_1])))))
let x111 = (imap2 16 64 (\x112_0 x112_1 -> (isum1 16 (\x113_0 -> (x108[x112_0][x113_0] F.* wdown[x113_0][x112_1])))))
let x114 = (imap2 16 64 (\x115_0 x115_1 -> ((indicatorp x58[x115_0][x115_1]) F.* x111[x115_0][x115_1])))
let x116 = (imap2 16 16 (\x117_0 x117_1 -> (isum1 64 (\x118_0 -> (x114[x117_0][x118_0] F.* wup[x118_0][x117_1])))))
let x119 = (let x120 = (imap2 16 16 (\x127_0 x127_1 -> (x49[x127_0][x127_1] F.* x49[x127_0][x127_1])))
in (let x121 = (imap1 16 (\x128_0 -> ((isum1 16 (\x129_0 -> x120[x128_0][x129_0])) F./ fromi64 16)))
in (let x122 = (imap1 16 (\x130_0 -> (F.sqrt (x121[x130_0] F.+ (one F./ fromi64 100000)))))
in (let x123 = (imap1 16 (\x131_0 -> (isum1 16 (\x132_0 -> (F.neg ((x116[x131_0][x132_0] F.* x49[x131_0][x132_0]) F.* (one F./ (x122[x131_0] F.* x122[x131_0]))))))))
in (let x124 = (imap1 16 (\x133_0 -> (x123[x133_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x121[x133_0] F.+ (one F./ fromi64 100000))))))))
in (let x125 = (imap2 16 16 (\x134_0 x134_1 -> (x124[x134_0] F./ fromi64 16)))
in (imap2 16 16 (\x126_0 x126_1 -> (x108[x126_0][x126_1] F.+ (((x116[x126_0][x126_1] F.* (one F./ x122[x126_0])) F.+ (x125[x126_0][x126_1] F.* x49[x126_0][x126_1])) F.+ (x125[x126_0][x126_1] F.* x49[x126_0][x126_1])))))))))))
let x135 = (imap2 16 16 (\x136_0 x136_1 -> (isum1 16 (\x137_0 -> (x119[x136_0][x137_0] F.* wout[x137_0][x136_1])))))
let x138 = (imap3 4 16 4 (\x139_0 x139_1 x139_2 -> x135[x139_1][((x139_0 * 4) + x139_2)]))
let x140 = (let x141 = (imap3 4 16 16 (\x143_0 x143_1 x143_2 -> (let x144 = (imaximum1 16 (\x147_0 -> (((isum1 4 (\x148_0 -> (x25[x143_0][x143_1][x148_0] F.* x27[x143_0][x147_0][x148_0]))) F./ fromi64 2) F.+ mask[x143_1][x147_0])))
in (let x145 = (imap1 16 (\x149_0 -> (F.exp ((((isum1 4 (\x150_0 -> (x25[x143_0][x143_1][x150_0] F.* x27[x143_0][x149_0][x150_0]))) F./ fromi64 2) F.+ mask[x143_1][x149_0]) F.+ (F.neg x144)))))
in (let x146 = (isum1 16 (\x151_0 -> x145[x151_0]))
in (x145[x143_2] F.* (one F./ x146)))))))
in (imap3 4 16 4 (\x142_0 x142_1 x142_2 -> (isum1 16 (\x152_0 -> (x138[x142_0][x152_0][x142_2] F.* x141[x142_0][x152_0][x142_1]))))))
let x153 = (let x154 = (imap3 4 16 16 (\x163_0 x163_1 x163_2 -> (isum1 4 (\x164_0 -> (x138[x163_0][x163_1][x164_0] F.* x29[x163_0][x163_2][x164_0])))))
in (let x155 = (imap2 4 16 (\x165_0 x165_1 -> (imaximum1 16 (\x166_0 -> (((isum1 4 (\x167_0 -> (x25[x165_0][x165_1][x167_0] F.* x27[x165_0][x166_0][x167_0]))) F./ fromi64 2) F.+ mask[x165_1][x166_0])))))
in (let x156 = (imap3 4 16 16 (\x168_0 x168_1 x168_2 -> (F.exp ((((isum1 4 (\x169_0 -> (x25[x168_0][x168_1][x169_0] F.* x27[x168_0][x168_2][x169_0]))) F./ fromi64 2) F.+ mask[x168_1][x168_2]) F.+ (F.neg x155[x168_0][x168_1])))))
in (let x157 = (imap2 4 16 (\x170_0 x170_1 -> (isum1 16 (\x171_0 -> x156[x170_0][x170_1][x171_0]))))
in (let x158 = (imap2 4 16 (\x172_0 x172_1 -> (isum1 16 (\x173_0 -> (F.neg ((x154[x172_0][x172_1][x173_0] F.* x156[x172_0][x172_1][x173_0]) F.* (one F./ (x157[x172_0][x172_1] F.* x157[x172_0][x172_1]))))))))
in (let x159 = (imap3 4 16 16 (\x174_0 x174_1 x174_2 -> ((x154[x174_0][x174_1][x174_2] F.* (one F./ x157[x174_0][x174_1])) F.+ x158[x174_0][x174_1])))
in (let x160 = (imap2 4 16 (\x175_0 x175_1 -> (isum1 16 (\x176_0 -> (F.neg ((F.exp ((((isum1 4 (\x177_0 -> (x25[x175_0][x175_1][x177_0] F.* x27[x175_0][x176_0][x177_0]))) F./ fromi64 2) F.+ mask[x175_1][x176_0]) F.+ (F.neg x155[x175_0][x175_1]))) F.* x159[x175_0][x175_1][x176_0]))))))
in (let x161 = (imap2 4 16 (\x178_0 x178_1 -> (one F./ (isum1 16 (\x179_0 -> (one F.+ (F.neg (indicatorp (F.neg ((((isum1 4 (\x180_0 -> (x25[x178_0][x178_1][x180_0] F.* x27[x178_0][x179_0][x180_0]))) F./ fromi64 2) F.+ mask[x178_1][x179_0]) F.+ (F.neg x155[x178_0][x178_1])))))))))))
in (imap3 4 16 4 (\x162_0 x162_1 x162_2 -> (isum1 16 (\x181_0 -> (((((F.exp ((((isum1 4 (\x183_0 -> (x25[x162_0][x181_0][x183_0] F.* x27[x162_0][x162_1][x183_0]))) F./ fromi64 2) F.+ mask[x181_0][x162_1]) F.+ (F.neg x155[x162_0][x181_0]))) F.* x159[x162_0][x181_0][x162_1]) F./ fromi64 2) F.* x25[x162_0][x181_0][x162_2]) F.+ ((((x160[x162_0][x181_0] F.* (one F.+ (F.neg (indicatorp (F.neg ((((isum1 4 (\x182_0 -> (x25[x162_0][x181_0][x182_0] F.* x27[x162_0][x162_1][x182_0]))) F./ fromi64 2) F.+ mask[x181_0][x162_1]) F.+ (F.neg x155[x162_0][x181_0]))))))) F.* x161[x162_0][x181_0]) F./ fromi64 2) F.* x25[x162_0][x181_0][x162_2]))))))))))))))
let x184 = (let x185 = (imap3 4 16 16 (\x194_0 x194_1 x194_2 -> (isum1 4 (\x195_0 -> (x138[x194_0][x194_1][x195_0] F.* x29[x194_0][x194_2][x195_0])))))
in (let x186 = (imap2 4 16 (\x196_0 x196_1 -> (imaximum1 16 (\x197_0 -> (((isum1 4 (\x198_0 -> (x25[x196_0][x196_1][x198_0] F.* x27[x196_0][x197_0][x198_0]))) F./ fromi64 2) F.+ mask[x196_1][x197_0])))))
in (let x187 = (imap3 4 16 16 (\x199_0 x199_1 x199_2 -> (F.exp ((((isum1 4 (\x200_0 -> (x25[x199_0][x199_1][x200_0] F.* x27[x199_0][x199_2][x200_0]))) F./ fromi64 2) F.+ mask[x199_1][x199_2]) F.+ (F.neg x186[x199_0][x199_1])))))
in (let x188 = (imap2 4 16 (\x201_0 x201_1 -> (isum1 16 (\x202_0 -> x187[x201_0][x201_1][x202_0]))))
in (let x189 = (imap2 4 16 (\x203_0 x203_1 -> (isum1 16 (\x204_0 -> (F.neg ((x185[x203_0][x203_1][x204_0] F.* x187[x203_0][x203_1][x204_0]) F.* (one F./ (x188[x203_0][x203_1] F.* x188[x203_0][x203_1]))))))))
in (let x190 = (imap3 4 16 16 (\x205_0 x205_1 x205_2 -> ((x185[x205_0][x205_1][x205_2] F.* (one F./ x188[x205_0][x205_1])) F.+ x189[x205_0][x205_1])))
in (let x191 = (imap2 4 16 (\x206_0 x206_1 -> (isum1 16 (\x207_0 -> (F.neg ((F.exp ((((isum1 4 (\x208_0 -> (x25[x206_0][x206_1][x208_0] F.* x27[x206_0][x207_0][x208_0]))) F./ fromi64 2) F.+ mask[x206_1][x207_0]) F.+ (F.neg x186[x206_0][x206_1]))) F.* x190[x206_0][x206_1][x207_0]))))))
in (let x192 = (imap2 4 16 (\x209_0 x209_1 -> (one F./ (isum1 16 (\x210_0 -> (one F.+ (F.neg (indicatorp (F.neg ((((isum1 4 (\x211_0 -> (x25[x209_0][x209_1][x211_0] F.* x27[x209_0][x210_0][x211_0]))) F./ fromi64 2) F.+ mask[x209_1][x210_0]) F.+ (F.neg x186[x209_0][x209_1])))))))))))
in (imap3 4 16 4 (\x193_0 x193_1 x193_2 -> ((isum1 16 (\x212_0 -> ((((F.exp ((((isum1 4 (\x215_0 -> (x25[x193_0][x193_1][x215_0] F.* x27[x193_0][x212_0][x215_0]))) F./ fromi64 2) F.+ mask[x193_1][x212_0]) F.+ (F.neg x186[x193_0][x193_1]))) F.* x190[x193_0][x193_1][x212_0]) F./ fromi64 2) F.* x27[x193_0][x212_0][x193_2]))) F.+ (isum1 16 (\x213_0 -> ((((x191[x193_0][x193_1] F.* (one F.+ (F.neg (indicatorp (F.neg ((((isum1 4 (\x214_0 -> (x25[x193_0][x193_1][x214_0] F.* x27[x193_0][x213_0][x214_0]))) F./ fromi64 2) F.+ mask[x193_1][x213_0]) F.+ (F.neg x186[x193_0][x193_1]))))))) F.* x192[x193_0][x193_1]) F./ fromi64 2) F.* x27[x193_0][x213_0][x193_2]))))))))))))))
let x216 = (imap2 16 16 (\x217_0 x217_1 -> x140[(x217_1 / 4)][x217_0][(x217_1 % 4)]))
let x218 = (imap2 16 16 (\x219_0 x219_1 -> x153[(x219_1 / 4)][x219_0][(x219_1 % 4)]))
let x220 = (imap2 16 16 (\x221_0 x221_1 -> x184[(x221_1 / 4)][x221_0][(x221_1 % 4)]))
let x222 = (imap2 16 16 (\x223_0 x223_1 -> (((isum1 16 (\x224_0 -> (x216[x223_0][x224_0] F.* wval[x224_0][x223_1]))) F.+ (isum1 16 (\x225_0 -> (x218[x223_0][x225_0] F.* wkey[x225_0][x223_1])))) F.+ (isum1 16 (\x226_0 -> (x220[x223_0][x226_0] F.* wqry[x226_0][x223_1]))))))
let x227 = (let x228 = (imap2 16 16 (\x235_0 x235_1 -> (x2[x235_0][x235_1] F.* x2[x235_0][x235_1])))
in (let x229 = (imap1 16 (\x236_0 -> ((isum1 16 (\x237_0 -> x228[x236_0][x237_0])) F./ fromi64 16)))
in (let x230 = (imap1 16 (\x238_0 -> (F.sqrt (x229[x238_0] F.+ (one F./ fromi64 100000)))))
in (let x231 = (imap1 16 (\x239_0 -> (isum1 16 (\x240_0 -> (F.neg ((x222[x239_0][x240_0] F.* x2[x239_0][x240_0]) F.* (one F./ (x230[x239_0] F.* x230[x239_0]))))))))
in (let x232 = (imap1 16 (\x241_0 -> (x231[x241_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x229[x241_0] F.+ (one F./ fromi64 100000))))))))
in (let x233 = (imap2 16 16 (\x242_0 x242_1 -> (x232[x242_0] F./ fromi64 16)))
in (imap2 16 16 (\x234_0 x234_1 -> (x119[x234_0][x234_1] F.+ (((x222[x234_0][x234_1] F.* (one F./ x230[x234_0])) F.+ (x233[x234_0][x234_1] F.* x2[x234_0][x234_1])) F.+ (x233[x234_0][x234_1] F.* x2[x234_0][x234_1])))))))))))
let x243 = (let x244 = (imap2 16 16 (\x251_0 x251_1 -> (x0[x251_0][x251_1] F.* x0[x251_0][x251_1])))
in (let x245 = (imap1 16 (\x252_0 -> ((isum1 16 (\x253_0 -> x244[x252_0][x253_0])) F./ fromi64 16)))
in (let x246 = (imap1 16 (\x254_0 -> (F.sqrt (x245[x254_0] F.+ (one F./ fromi64 100000)))))
in (let x247 = (imap1 16 (\x255_0 -> (isum1 16 (\x256_0 -> (F.neg ((x227[x255_0][x256_0] F.* x0[x255_0][x256_0]) F.* (one F./ (x246[x255_0] F.* x246[x255_0]))))))))
in (let x248 = (imap1 16 (\x257_0 -> (x247[x257_0] F.* (one F./ ((one F.+ one) F.* (F.sqrt (x245[x257_0] F.+ (one F./ fromi64 100000))))))))
in (let x249 = (imap2 16 16 (\x258_0 x258_1 -> (x248[x258_0] F./ fromi64 16)))
in (imap2 16 16 (\x250_0 x250_1 -> (((x227[x250_0][x250_1] F.* (one F./ x246[x250_0])) F.+ (x249[x250_0][x250_1] F.* x0[x250_0][x250_1])) F.+ (x249[x250_0][x250_1] F.* x0[x250_0][x250_1]))))))))))

let dmask = (let x259 = (imap3 4 16 16 (\x268_0 x268_1 x268_2 -> (isum1 4 (\x269_0 -> (x138[x268_0][x268_1][x269_0] F.* x29[x268_0][x268_2][x269_0])))))
in (let x260 = (imap2 4 16 (\x270_0 x270_1 -> (imaximum1 16 (\x271_0 -> (((isum1 4 (\x272_0 -> (x25[x270_0][x270_1][x272_0] F.* x27[x270_0][x271_0][x272_0]))) F./ fromi64 2) F.+ mask[x270_1][x271_0])))))
in (let x261 = (imap3 4 16 16 (\x273_0 x273_1 x273_2 -> (F.exp ((((isum1 4 (\x274_0 -> (x25[x273_0][x273_1][x274_0] F.* x27[x273_0][x273_2][x274_0]))) F./ fromi64 2) F.+ mask[x273_1][x273_2]) F.+ (F.neg x260[x273_0][x273_1])))))
in (let x262 = (imap2 4 16 (\x275_0 x275_1 -> (isum1 16 (\x276_0 -> x261[x275_0][x275_1][x276_0]))))
in (let x263 = (imap2 4 16 (\x277_0 x277_1 -> (isum1 16 (\x278_0 -> (F.neg ((x259[x277_0][x277_1][x278_0] F.* x261[x277_0][x277_1][x278_0]) F.* (one F./ (x262[x277_0][x277_1] F.* x262[x277_0][x277_1]))))))))
in (let x264 = (imap3 4 16 16 (\x279_0 x279_1 x279_2 -> ((x259[x279_0][x279_1][x279_2] F.* (one F./ x262[x279_0][x279_1])) F.+ x263[x279_0][x279_1])))
in (let x265 = (imap2 4 16 (\x280_0 x280_1 -> (isum1 16 (\x281_0 -> (F.neg ((F.exp ((((isum1 4 (\x282_0 -> (x25[x280_0][x280_1][x282_0] F.* x27[x280_0][x281_0][x282_0]))) F./ fromi64 2) F.+ mask[x280_1][x281_0]) F.+ (F.neg x260[x280_0][x280_1]))) F.* x264[x280_0][x280_1][x281_0]))))))
in (let x266 = (imap2 4 16 (\x283_0 x283_1 -> (one F./ (isum1 16 (\x284_0 -> (one F.+ (F.neg (indicatorp (F.neg ((((isum1 4 (\x285_0 -> (x25[x283_0][x283_1][x285_0] F.* x27[x283_0][x284_0][x285_0]))) F./ fromi64 2) F.+ mask[x283_1][x284_0]) F.+ (F.neg x260[x283_0][x283_1])))))))))))
in (imap2 16 16 (\x267_0 x267_1 -> (isum1 4 (\x286_0 -> (((F.exp ((((isum1 4 (\x288_0 -> (x25[x286_0][x267_0][x288_0] F.* x27[x286_0][x267_1][x288_0]))) F./ fromi64 2) F.+ mask[x267_0][x267_1]) F.+ (F.neg x260[x286_0][x267_0]))) F.* x264[x286_0][x267_0][x267_1]) F.+ ((x265[x286_0][x267_0] F.* (one F.+ (F.neg (indicatorp (F.neg ((((isum1 4 (\x287_0 -> (x25[x286_0][x267_0][x287_0] F.* x27[x286_0][x267_1][x287_0]))) F./ fromi64 2) F.+ mask[x267_0][x267_1]) F.+ (F.neg x260[x286_0][x267_0]))))))) F.* x266[x286_0][x267_0]))))))))))))))
let dwpe = (imap2 16 16 (\x289_0 x289_1 -> x243[x289_0][x289_1]))
let dwqry = (imap2 16 16 (\x290_0 x290_1 -> (isum1 16 (\x291_0 -> (x220[x291_0][x290_0] F.* x9[x291_0][x290_1])))))
let dwkey = (imap2 16 16 (\x292_0 x292_1 -> (isum1 16 (\x293_0 -> (x218[x293_0][x292_0] F.* x9[x293_0][x292_1])))))
let dwval = (imap2 16 16 (\x294_0 x294_1 -> (isum1 16 (\x295_0 -> (x216[x295_0][x294_0] F.* x9[x295_0][x294_1])))))
let dwout = (imap2 16 16 (\x296_0 x296_1 -> (isum1 16 (\x297_0 -> (x119[x297_0][x296_0] F.* x44[x297_0][x296_1])))))
let dwup = (imap2 64 16 (\x298_0 x298_1 -> (isum1 16 (\x299_0 -> (x114[x299_0][x298_0] F.* x51[x299_0][x298_1])))))
let dwdown = (imap2 16 64 (\x300_0 x300_1 -> (isum1 16 (\x301_0 -> (x108[x301_0][x300_0] F.* x61[x301_0][x300_1])))))
let dwvoc = (imap2 27 16 (\x302_0 x302_1 -> (isum1 16 (\x303_0 -> (x73[x303_0][x302_0] F.* x66[x303_0][x302_1])))))
let dwseq = (imap2 16 16 (\x304_0 x304_1 -> x243[x304_0][x304_1]))
let dtarget = (let x305 = (imap2 16 27 (\x307_0 x307_1 -> (let x308 = (imaximum1 27 (\x311_0 -> x68[x307_0][x311_0]))
in (let x309 = (imap1 27 (\x312_0 -> (F.exp (x68[x307_0][x312_0] F.+ (F.neg x308)))))
in (let x310 = (isum1 27 (\x313_0 -> x309[x313_0]))
in (F.log (x309[x307_1] F.* (one F./ x310))))))))
in (imap2 16 27 (\x306_0 x306_1 -> ((F.neg x71[x306_0]) F.* x305[x306_0][x306_1]))))

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