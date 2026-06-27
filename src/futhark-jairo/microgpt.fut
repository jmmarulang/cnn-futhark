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

    (let x0 = (imap2 16 16 (\x19_0 x19_1 -> ((wpe[x19_0][x19_1] F.+ wseq[x19_0][x19_1]) F.* (one F./ (F.sqrt (((isum1 16 (\x20_0 -> ((wpe[x19_0][x20_0] F.+ wseq[x19_0][x20_0]) F.* (wpe[x19_0][x20_0] F.+ wseq[x19_0][x20_0])))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))))))
    in (let x1 = (imap2 16 16 (\x21_0 x21_1 -> (x0[x21_0][x21_1] F.* (one F./ (F.sqrt (((isum1 16 (\x22_0 -> (x0[x21_0][x22_0] F.* x0[x21_0][x22_0]))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))))))
    in (let x2 = (imap2 16 16 (\x23_0 x23_1 -> (isum1 16 (\x24_0 -> (wqry[x23_1][x24_0] F.* x1[x23_0][x24_0])))))
    in (let x3 = (imap2 16 16 (\x25_0 x25_1 -> (isum1 16 (\x26_0 -> (wkey[x25_1][x26_0] F.* x1[x25_0][x26_0])))))
    in (let x4 = (imap2 16 16 (\x27_0 x27_1 -> (isum1 16 (\x28_0 -> (wval[x27_1][x28_0] F.* x1[x27_0][x28_0])))))
    in (let x5 = (imap3 4 16 4 (\x29_0 x29_1 x29_2 -> x2[x29_1][((x29_0 * 4) + x29_2)]))
    in (let x6 = (imap3 4 16 4 (\x30_0 x30_1 x30_2 -> x3[x30_1][((x30_0 * 4) + x30_2)]))
    in (let x7 = (imap3 4 16 4 (\x31_0 x31_1 x31_2 -> x4[x31_1][((x31_0 * 4) + x31_2)]))
    in (let x8 = (imap3 4 16 4 (\x32_0 x32_1 x32_2 -> (isum1 16 (\x33_0 -> (((F.exp (((isum1 4 (\x35_0 -> (x5[x32_0][x32_1][x35_0] F.* x6[x32_0][x33_0][x35_0]))) F./ fromi64 2) F.+ mask[x32_1][x33_0])) F.* (one F./ (isum1 16 (\x34_0 -> (F.exp (((isum1 4 (\x36_0 -> (x5[x32_0][x32_1][x36_0] F.* x6[x32_0][x34_0][x36_0]))) F./ fromi64 2) F.+ mask[x32_1][x34_0])))))) F.* x7[x32_0][x33_0][x32_2])))))
    in (let x9 = (imap2 16 16 (\x37_0 x37_1 -> x8[(x37_1 / 4)][x37_0][(x37_1 % 4)]))
    in (let x10 = (imap2 16 16 (\x38_0 x38_1 -> (isum1 16 (\x39_0 -> (wout[x38_1][x39_0] F.* x9[x38_0][x39_0])))))
    in (let x11 = (imap2 16 16 (\x40_0 x40_1 -> (x10[x40_0][x40_1] F.+ x0[x40_0][x40_1])))
    in (let x12 = (imap2 16 16 (\x41_0 x41_1 -> (x11[x41_0][x41_1] F.* (one F./ (F.sqrt (((isum1 16 (\x42_0 -> (x11[x41_0][x42_0] F.* x11[x41_0][x42_0]))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))))))
    in (let x13 = (imap2 16 64 (\x43_0 x43_1 -> (isum1 16 (\x44_0 -> (wup[x43_1][x44_0] F.* x12[x43_0][x44_0])))))
    in (let x14 = (imap2 16 64 (\x45_0 x45_1 -> F.max x13[x45_0][x45_1] zero))
    in (let x15 = (imap2 16 16 (\x46_0 x46_1 -> (isum1 64 (\x47_0 -> (wdown[x46_1][x47_0] F.* x14[x46_0][x47_0])))))
    in (let x16 = (imap2 16 16 (\x48_0 x48_1 -> (x15[x48_0][x48_1] F.+ x11[x48_0][x48_1])))
    in (let x17 = (imap2 16 27 (\x49_0 x49_1 -> (isum1 16 (\x50_0 -> (wvoc[x49_1][x50_0] F.* x16[x49_0][x50_0])))))
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

    (let x0 = (imap2 16 16 (\x20_0 x20_1 -> ((wpe[x20_0][x20_1] F.+ wseq[x20_0][x20_1]) F.* (one F./ (F.sqrt (((isum1 16 (\x21_0 -> ((wpe[x20_0][x21_0] F.+ wseq[x20_0][x21_0]) F.* (wpe[x20_0][x21_0] F.+ wseq[x20_0][x21_0])))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))))))
    in (let x1 = (imap2 16 16 (\x22_0 x22_1 -> (x0[x22_0][x22_1] F.* (one F./ (F.sqrt (((isum1 16 (\x23_0 -> (x0[x22_0][x23_0] F.* x0[x22_0][x23_0]))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))))))
    in (let x2 = (imap2 16 16 (\x24_0 x24_1 -> (isum1 16 (\x25_0 -> (wqry[x24_1][x25_0] F.* x1[x24_0][x25_0])))))
    in (let x3 = (imap2 16 16 (\x26_0 x26_1 -> (isum1 16 (\x27_0 -> (wkey[x26_1][x27_0] F.* x1[x26_0][x27_0])))))
    in (let x4 = (imap2 16 16 (\x28_0 x28_1 -> (isum1 16 (\x29_0 -> (wval[x28_1][x29_0] F.* x1[x28_0][x29_0])))))
    in (let x5 = (imap3 4 16 4 (\x30_0 x30_1 x30_2 -> x2[x30_1][((x30_0 * 4) + x30_2)]))
    in (let x6 = (imap3 4 16 4 (\x31_0 x31_1 x31_2 -> x3[x31_1][((x31_0 * 4) + x31_2)]))
    in (let x7 = (imap3 4 16 4 (\x32_0 x32_1 x32_2 -> x4[x32_1][((x32_0 * 4) + x32_2)]))
    in (let x8 = (imap3 4 16 4 (\x33_0 x33_1 x33_2 -> (isum1 16 (\x34_0 -> (((F.exp (((isum1 4 (\x36_0 -> (x5[x33_0][x33_1][x36_0] F.* x6[x33_0][x34_0][x36_0]))) F./ fromi64 2) F.+ mask[x33_1][x34_0])) F.* (one F./ (isum1 16 (\x35_0 -> (F.exp (((isum1 4 (\x37_0 -> (x5[x33_0][x33_1][x37_0] F.* x6[x33_0][x35_0][x37_0]))) F./ fromi64 2) F.+ mask[x33_1][x35_0])))))) F.* x7[x33_0][x34_0][x33_2])))))
    in (let x9 = (imap2 16 16 (\x38_0 x38_1 -> x8[(x38_1 / 4)][x38_0][(x38_1 % 4)]))
    in (let x10 = (imap2 16 16 (\x39_0 x39_1 -> (isum1 16 (\x40_0 -> (wout[x39_1][x40_0] F.* x9[x39_0][x40_0])))))
    in (let x11 = (imap2 16 16 (\x41_0 x41_1 -> (x10[x41_0][x41_1] F.+ x0[x41_0][x41_1])))
    in (let x12 = (imap2 16 16 (\x42_0 x42_1 -> (x11[x42_0][x42_1] F.* (one F./ (F.sqrt (((isum1 16 (\x43_0 -> (x11[x42_0][x43_0] F.* x11[x42_0][x43_0]))) F./ fromi64 16) F.+ (one F./ fromi64 100000)))))))
    in (let x13 = (imap2 16 64 (\x44_0 x44_1 -> (isum1 16 (\x45_0 -> (wup[x44_1][x45_0] F.* x12[x44_0][x45_0])))))
    in (let x14 = (imap2 16 64 (\x46_0 x46_1 -> F.max x13[x46_0][x46_1] zero))
    in (let x15 = (imap2 16 16 (\x47_0 x47_1 -> (isum1 64 (\x48_0 -> (wdown[x47_1][x48_0] F.* x14[x47_0][x48_0])))))
    in (let x16 = (imap2 16 16 (\x49_0 x49_1 -> (x15[x49_0][x49_1] F.+ x11[x49_0][x49_1])))
    in (let x17 = (imap2 16 27 (\x50_0 x50_1 -> (isum1 16 (\x51_0 -> (wvoc[x50_1][x51_0] F.* x16[x50_0][x51_0])))))
    in (let x18 = (imap1 16 (\x52_0 -> (F.neg (isum1 27 (\x53_0 -> ((F.log ((F.exp x17[x52_0][x53_0]) F.* (one F./ (isum1 27 (\x54_0 -> (F.exp x17[x52_0][x54_0])))))) F.* target[x52_0][x53_0]))))))
    in (let x19 = ((isum1 16 (\x55_0 -> x18[x55_0])) F./ fromi64 16)
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
    -- #[unsafe]
    \(mask: [16][16]real) (wpe: [16][16]real)
    (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real)
    (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real)
    (wvoc: [27][16]real) (wseq: [16][16]real) (target: [16][27]real) ->
    -- (wqry, wqry, wqry, wqry, wqry, wup, wdown, wvoc, wseq)
-- Let seq := m-rmsnorm {sl} ((p .wpe) ⊞ wseq) In
let x0 = (imap2 16 16 (\x1_0 x1_1 -> (wpe[x1_0][x1_1] F.+ wseq[x1_0][x1_1])))
-- Let nseq := m-rmsnorm {sl} seq In
let x2 = (imap2 16 16 (\x3_0 x3_1 -> x0[x3_0][x3_1]))
-- Let qs := m-linear {u = ed} ⟨ p .wqry ⟩ nseq In ?
let x4 = (imap3 4 16 4 (\x5_0 x5_1 x5_2 -> (isum1 16 (\x6_0 -> (wqry[((x5_0 * 4) + x5_2)][x6_0] F.* x2[x5_1][x6_0])))))
-- Let ks := m-linear {u = ed} ⟨ p .wkey ⟩ nseq In
let x7 = (imap3 4 16 4 (\x8_0 x8_1 x8_2 -> (isum1 16 (\x9_0 -> (wkey[((x8_0 * 4) + x8_2)][x9_0] F.* x2[x8_1][x9_0])))))
-- Let vs := m-linear {u = ed} ⟨ p .wval ⟩ nseq In ?
let x10 = (imap3 4 16 4 (\x11_0 x11_1 x11_2 -> (isum1 16 (\x12_0 -> (wval[((x11_0 * 4) + x11_2)][x12_0] F.* x2[x11_1][x12_0])))))
-- Let attn := unblock-vec battn pr In ?
let x13 = (imap2 16 16 (\x14_0 x14_1 -> (isum1 16 (\x15_0 -> ((((isum1 4 (\x16_0 -> (x4[(x14_1 / 4)][x14_0][x16_0] F.* x7[(x14_1 / 4)][x15_0][x16_0]))) F./ fromi64 2) F.+ mask[x14_0][x15_0]) F.* x10[(x14_1 / 4)][x15_0][(x14_1 % 4)])))))
-- Let oseq := m-linear {u = ed} {p = sl} ⟨ p .wout ⟩ attn In ?
let x17 = (imap2 16 16 (\x18_0 x18_1 -> ((isum1 16 (\x19_0 -> (wout[x18_1][x19_0] F.* x13[x18_0][x19_0]))) F.+ x0[x18_0][x18_1])))
-- Let nseq2 := m-rmsnorm {sl} cseq In ?
let x20 = (imap2 16 16 (\x21_0 x21_1 -> x17[x21_0][x21_1]))
-- Let useq := m-linear {p = sl} ⟨ p .wup ⟩ nseq2 In ?
let x22 = (imap2 16 64 (\x23_0 x23_1 -> (isum1 16 (\x24_0 -> (wup[x23_1][x24_0] F.* x20[x23_0][x24_0])))))
-- Let aseq := relu useq In
let x25 = (imap2 16 64 (\x26_0 x26_1 -> F.max x22[x26_0][x26_1] zero))
-- Let dseq := m-linear {u = ed} {p = sl} ⟨ p .wdown ⟩ aseq In
let x27 = (imap2 16 16 (\x28_0 x28_1 -> ((isum1 64 (\x29_0 -> (wdown[x28_1][x29_0] F.* x25[x28_0][x29_0]))) F.+ x17[x28_0][x28_1])))
-- Let logits := m-linear {u = vo} {p = sl} ⟨ p .wvoc ⟩ lseq In ?
let x30 = (imap2 16 27 (\x31_0 x31_1 -> (isum1 27 (\x32_0 -> ((if ((x31_1 == x32_0)) then (F.neg (one F./ fromi64 16)) else zero) F.* (one F./ (isum1 16 (\x33_0 -> (wvoc[x31_1][x33_0] F.* x27[x31_0][x33_0])))))))))
-- Let logits := m-linear {u = vo} {p = sl} ⟨ p .wvoc ⟩ lseq In ?
let x34 = (imap2 16 16 (\x35_0 x35_1 -> (isum1 27 (\x36_0 -> (x30[x35_0][x36_0] F.* wvoc[x36_0][x35_1])))))
-- d relu
let x37 = (imap2 16 64 (\x38_0 x38_1 -> (indicatorp x22[x38_0][x38_1] F.* (isum1 16 (\x39_0 -> (x34[x38_0][x39_0] F.* wdown[x39_0][x38_1]))))))
-- ?
let x40 = (imap2 16 16 (\x41_0 x41_1 -> (x34[x41_0][x41_1] F.+ (isum1 64 (\x42_0 -> (x37[x41_0][x42_0] F.* wup[x42_0][x41_1]))))))
-- adj bqs ?
let x43 = (imap3 4 16 4 (\x44_0 x44_1 x44_2 -> (isum1 16 (\x45_0 -> (isum1 4 (\x46_0 -> (if ((x44_1 == x45_0)) then (if ((x44_0 == x46_0)) then (isum1 16 (\x47_0 -> (x40[x45_0][x47_0] F.* wout[x47_0][((x46_0 * 4) + x44_2)]))) else zero) else zero)))))))
-- d attn?
-- let x48 = (imap2 16 16 (\x49_0 x49_1 -> (isum1 4 (\x50_0 -> (isum1 16 (\x51_0 -> (if ((x49_0 == x51_0)) then (if (((x49_1 / 4) == x50_0)) then (isum1 16 (\x52_0 -> (x43[x50_0][x52_0][(x49_1 % 4)] F.* (((isum1 4 (\x53_0 -> (x4[x50_0][x52_0][x53_0] F.* x7[x50_0][x51_0][x53_0]))) F./ fromi64 2) F.+ mask[x52_0][x51_0])))) else zero) else zero)))))))
-- 
-- let x54 = (imap2 16 16 (\x55_0 x55_1 -> (isum1 4 (\x56_0 -> (isum1 16 (\x57_0 -> (if ((x55_0 == x57_0)) then (if (((x55_1 / 4) == x56_0)) then (isum1 16 (\x58_0 -> (isum1 4 (\x59_0 -> (isum1 16 (\x60_0 -> (isum1 16 (\x61_0 -> (isum1 16 (\x62_0 -> (((if ((x62_0 == x61_0)) then (if ((x61_0 == x58_0)) then (if ((x57_0 == x60_0)) then (x43[x56_0][x58_0][x59_0] F.* x10[x56_0][x60_0][x59_0]) else zero) else zero) else zero) F./ fromi64 2) F.* x4[x56_0][x62_0][(x55_1 % 4)]))))))))))) else zero) else zero)))))))
-- adj qs ?
let x63 = (imap2 16 16 (\x64_0 x64_1 -> (isum1 4 (\x65_0 -> (isum1 16 (\x66_0 -> (if ((x64_0 == x66_0)) then (if (((x64_1 / 4) == x65_0)) then (isum1 16 (\x67_0 -> (isum1 4 (\x68_0 -> (isum1 16 (\x69_0 -> (isum1 16 (\x70_0 -> (isum1 16 (\x71_0 -> (((if ((x66_0 == x70_0)) then (if ((x70_0 == x67_0)) then (if ((x71_0 == x69_0)) then (x43[x65_0][x67_0][x68_0] F.* x10[x65_0][x69_0][x68_0]) else zero) else zero) else zero) F./ fromi64 2) F.* x7[x65_0][x71_0][(x64_1 % 4)]))))))))))) else zero) else zero)))))))
-- let x72 = (imap2 16 16 (\x73_0 x73_1 -> (x40[x73_0][x73_1] F.+ (((isum1 16 (\x74_0 -> (x48[x73_0][x74_0] F.* wval[x74_0][x73_1]))) F.+ (isum1 16 (\x75_0 -> (x54[x73_0][x75_0] F.* wkey[x75_0][x73_1])))) F.+ (isum1 16 (\x76_0 -> (x63[x73_0][x76_0] F.* wqry[x76_0][x73_1])))))))

-- -- let dmask = (imap2 16 16 (\x77_0 x77_1 -> (isum1 4 (\x78_0 -> (isum1 4 (\x79_0 -> (x43[x78_0][x77_0][x79_0] F.* x10[x78_0][x77_1][x79_0])))))))
-- let dwpe = (imap2 16 16 (\x80_0 x80_1 -> x72[x80_0][x80_1]))
-- let dwqry = (imap2 16 16 (\x81_0 x81_1 -> (isum1 16 (\x82_0 -> (x63[x82_0][x81_0] F.* x2[x82_0][x81_1])))))
-- let dwkey = (imap2 16 16 (\x83_0 x83_1 -> (isum1 16 (\x84_0 -> (x54[x84_0][x83_0] F.* x2[x84_0][x83_1])))))
-- let dwval = (imap2 16 16 (\x85_0 x85_1 -> (isum1 16 (\x86_0 -> (x48[x86_0][x85_0] F.* x2[x86_0][x85_1])))))
-- let dwout = (imap2 16 16 (\x87_0 x87_1 -> (isum1 16 (\x88_0 -> (x40[x88_0][x87_0] F.* x13[x88_0][x87_1])))))
-- let dwup = (imap2 64 16 (\x89_0 x89_1 -> (isum1 16 (\x90_0 -> (x37[x90_0][x89_0] F.* x20[x90_0][x89_1])))))
-- let dwdown = (imap2 16 64 (\x91_0 x91_1 -> (isum1 16 (\x92_0 -> (x34[x92_0][x91_0] F.* x25[x92_0][x91_1])))))
-- let dwvoc = (imap2 27 16 (\x93_0 x93_1 -> (isum1 16 (\x94_0 -> (x30[x94_0][x93_0] F.* x27[x94_0][x93_1])))))
-- let dwseq = (imap2 16 16 (\x95_0 x95_1 -> x72[x95_0][x95_1]))
-- -- let dtarget = (imap2 16 27 (\x96_0 x96_1 -> zero))



  -- in (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq)
    in (x63, x63, wqry, wval, wout, wup, wdown, wvoc, wseq)
    -- with no sf, norm or target, x54 is the bottleneck
    -- in (imap2 16 16 (\n _ -> x38[0][0]),
    -- wkey, wkey, wval, wout, wup, x30, wvoc, wseq)
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

