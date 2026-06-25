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
    -- #[unsafe]
    \(mask: [16][16]real) (wpe: [16][16]real)
    (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real)
    (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real)
    (wvoc: [27][16]real) (wseq: [16][16]real) -> --(imap2 16 27 (\n m -> one F./ zero))

    (let x0 = (imap2 16 16 (\x5_0 x5_1 -> (let x6 = ((isum1 16 (\x9_0 -> ((wpe[x5_0][x9_0] F.+ wseq[x5_0][x9_0]) F.* (wpe[x5_0][x9_0] F.+ wseq[x5_0][x9_0])))) F./ fromi64 16)
    in (let x7 = (one F./ (F.sqrt (x6 F.+ (one F./ fromi64 100000))))
    in (let x8 = (imap1 16 (\x10_0 -> ((wpe[x5_0][x10_0] F.+ wseq[x5_0][x10_0]) F.* x7)))
    in x8[x5_1])))))
    in (let x1 = (imap3 16 4 4 (\x11_0 x11_1 x11_2 -> x0[x11_0][((x11_1 * 4) + x11_2)]))
    in (let x2 = (let x12 = (imap3 16 4 4 (\x21_0 x21_1 x21_2 -> (let x22 = ((isum2 4 4 (\x25_0 x25_1 -> (x1[x21_0][x25_0][x25_1] F.* x1[x21_0][x25_0][x25_1]))) F./ fromi64 16)
    in (let x23 = (one F./ (F.sqrt (x22 F.+ (one F./ fromi64 100000))))
    in (let x24 = (imap2 4 4 (\x26_0 x26_1 -> (x1[x21_0][x26_0][x26_1] F.* x23)))
    in x24[x21_1][x21_2])))))
    in (let x13 = (imap3 16 4 4 (\x27_0 x27_1 x27_2 -> (isum2 4 4 (\x28_0 x28_1 -> (wqry[((x27_1 * 4) + x27_2)][((x28_0 * 4) + x28_1)] F.* x12[x27_0][x28_0][x28_1])))))
    in (let x14 = (imap3 16 4 4 (\x29_0 x29_1 x29_2 -> (isum2 4 4 (\x30_0 x30_1 -> (wkey[((x29_1 * 4) + x29_2)][((x30_0 * 4) + x30_1)] F.* x12[x29_0][x30_0][x30_1])))))
    in (let x16 = (imap3 16 4 4 (\x31_0 x31_1 x31_2 -> (isum2 4 4 (\x32_0 x32_1 -> (wval[((x31_1 * 4) + x31_2)][((x32_0 * 4) + x32_1)] F.* x12[x31_0][x32_0][x32_1])))))
    in (let x16 = (imap3 16 4 4 (\x33_0 x33_1 x33_2 -> (let x34 = (imap2 16 16 (\x39_0 x39_1 -> (isum1 4 (\x40_0 -> (x13[x39_0][x33_1][x40_0] F.* x14[x39_1][x33_1][x40_0])))))
    in (let x35 = (imap2 16 16 (\x41_0 x41_1 -> (x34[x41_0][x41_1] F./ fromi64 2)))
    in (let x36 = (imap2 16 16 (\x42_0 x42_1 -> (x35[x42_0][x42_1] F.+ mask[x42_0][x42_1])))
    in (let x37 = (imap2 16 16 (\x43_0 x43_1 -> (let x44 = (imap1 16 (\x47_0 -> (F.exp x36[x43_0][x47_0])))
    in (let x45 = (isum1 16 (\x48_0 -> x44[x48_0]))
    in (let x46 = (imap1 16 (\x49_0 -> (x44[x49_0] F.* (one F./ x45))))
    in x46[x43_1])))))
    in (let x38 = (imap2 16 4 (\x50_0 x50_1 -> (isum1 16 (\x51_0 -> (x37[x50_0][x51_0] F.* x16[x51_0][x33_1][x50_1])))))
    in x38[x33_0][x33_2])))))))
    in (let x17 = (imap3 16 4 4 (\x52_0 x52_1 x52_2 -> (isum2 4 4 (\x53_0 x53_1 -> (wout[((x52_1 * 4) + x52_2)][((x53_0 * 4) + x53_1)] F.* x16[x52_0][x53_0][x53_1])))))
    in (let x18 = (imap3 16 4 4 (\x54_0 x54_1 x54_2 -> (x17[x54_0][x54_1][x54_2] F.+ x1[x54_0][x54_1][x54_2])))
    in (let x19 = (let x55 = (imap3 16 4 4 (\x61_0 x61_1 x61_2 -> (let x62 = ((isum2 4 4 (\x65_0 x65_1 -> (x18[x61_0][x65_0][x65_1] F.* x18[x61_0][x65_0][x65_1]))) F./ fromi64 16)
    in (let x63 = (one F./ (F.sqrt (x62 F.+ (one F./ fromi64 100000))))
    in (let x64 = (imap2 4 4 (\x66_0 x66_1 -> (x18[x61_0][x66_0][x66_1] F.* x63)))
    in x64[x61_1][x61_2])))))
    in (let x56 = (imap2 16 64 (\x67_0 x67_1 -> (isum2 4 4 (\x68_0 x68_1 -> (wup[x67_1][((x68_0 * 4) + x68_1)] F.* x55[x67_0][x68_0][x68_1])))))
    in (let x57 = (imap2 16 64 (\x69_0 x69_1 -> (if (zero F.<= x56[x69_0][x69_1]) then x56[x69_0][x69_1] else zero)))
    in (let x58 = (imap3 16 4 4 (\x70_0 x70_1 x70_2 -> (isum1 64 (\x71_0 -> (wdown[((x70_1 * 4) + x70_2)][x71_0] F.* x57[x70_0][x71_0])))))
    in (let x59 = (imap3 16 4 4 (\x72_0 x72_1 x72_2 -> (x58[x72_0][x72_1][x72_2] F.+ x18[x72_0][x72_1][x72_2])))
    in (imap3 16 4 4 (\x60_0 x60_1 x60_2 -> x59[x60_0][x60_1][x60_2])))))))
    in (imap3 16 4 4 (\x20_0 x20_1 x20_2 -> x19[x20_0][x20_1][x20_2]))))))))))
    in (let x3 = (imap2 16 27 (\x73_0 x73_1 -> (isum2 4 4 (\x74_0 x74_1 -> (wvoc[x73_1][((x74_0 * 4) + x74_1)] F.* x2[x73_0][x74_0][x74_1])))))
    in (imap2 16 27 (\x4_0 x4_1 -> x3[x4_0][x4_1]))))))

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

    (let x0 = (let x3 = (imap2 16 16 (\x8_0 x8_1 -> (let x9 = ((isum1 16 (\x12_0 -> ((wpe[x8_0][x12_0] F.+ wseq[x8_0][x12_0]) F.* (wpe[x8_0][x12_0] F.+ wseq[x8_0][x12_0])))) F./ fromi64 16)
    in (let x10 = (one F./ (F.sqrt (x9 F.+ (one F./ fromi64 100000))))
    in (let x11 = (imap1 16 (\x13_0 -> ((wpe[x8_0][x13_0] F.+ wseq[x8_0][x13_0]) F.* x10)))
    in x11[x8_1])))))
    in (let x4 = (imap3 16 4 4 (\x14_0 x14_1 x14_2 -> x3[x14_0][((x14_1 * 4) + x14_2)]))
    in (let x5 = (let x16 = (imap3 16 4 4 (\x24_0 x24_1 x24_2 -> (let x25 = ((isum2 4 4 (\x28_0 x28_1 -> (x4[x24_0][x28_0][x28_1] F.* x4[x24_0][x28_0][x28_1]))) F./ fromi64 16)
    in (let x26 = (one F./ (F.sqrt (x25 F.+ (one F./ fromi64 100000))))
    in (let x27 = (imap2 4 4 (\x29_0 x29_1 -> (x4[x24_0][x29_0][x29_1] F.* x26)))
    in x27[x24_1][x24_2])))))
    in (let x16 = (imap3 16 4 4 (\x30_0 x30_1 x30_2 -> (isum2 4 4 (\x31_0 x31_1 -> (wqry[((x30_1 * 4) + x30_2)][((x31_0 * 4) + x31_1)] F.* x16[x30_0][x31_0][x31_1])))))
    in (let x17 = (imap3 16 4 4 (\x32_0 x32_1 x32_2 -> (isum2 4 4 (\x33_0 x33_1 -> (wkey[((x32_1 * 4) + x32_2)][((x33_0 * 4) + x33_1)] F.* x16[x32_0][x33_0][x33_1])))))
    in (let x18 = (imap3 16 4 4 (\x34_0 x34_1 x34_2 -> (isum2 4 4 (\x35_0 x35_1 -> (wval[((x34_1 * 4) + x34_2)][((x35_0 * 4) + x35_1)] F.* x16[x34_0][x35_0][x35_1])))))
    in (let x19 = (imap3 16 4 4 (\x36_0 x36_1 x36_2 -> (let x37 = (imap2 16 16 (\x42_0 x42_1 -> (isum1 4 (\x43_0 -> (x16[x42_0][x36_1][x43_0] F.* x17[x42_1][x36_1][x43_0])))))
    in (let x38 = (imap2 16 16 (\x44_0 x44_1 -> (x37[x44_0][x44_1] F./ fromi64 2)))
    in (let x39 = (imap2 16 16 (\x45_0 x45_1 -> (x38[x45_0][x45_1] F.+ mask[x45_0][x45_1])))
    in (let x40 = (imap2 16 16 (\x46_0 x46_1 -> (let x47 = (imap1 16 (\x50_0 -> (F.exp x39[x46_0][x50_0])))
    in (let x48 = (isum1 16 (\x51_0 -> x47[x51_0]))
    in (let x49 = (imap1 16 (\x52_0 -> (x47[x52_0] F.* (one F./ x48))))
    in x49[x46_1])))))
    in (let x41 = (imap2 16 4 (\x53_0 x53_1 -> (isum1 16 (\x54_0 -> (x40[x53_0][x54_0] F.* x18[x54_0][x36_1][x53_1])))))
    in x41[x36_0][x36_2])))))))
    in (let x20 = (imap3 16 4 4 (\x55_0 x55_1 x55_2 -> (isum2 4 4 (\x56_0 x56_1 -> (wout[((x55_1 * 4) + x55_2)][((x56_0 * 4) + x56_1)] F.* x19[x55_0][x56_0][x56_1])))))
    in (let x21 = (imap3 16 4 4 (\x57_0 x57_1 x57_2 -> (x20[x57_0][x57_1][x57_2] F.+ x4[x57_0][x57_1][x57_2])))
    in (let x22 = (let x58 = (imap3 16 4 4 (\x64_0 x64_1 x64_2 -> (let x65 = ((isum2 4 4 (\x68_0 x68_1 -> (x21[x64_0][x68_0][x68_1] F.* x21[x64_0][x68_0][x68_1]))) F./ fromi64 16)
    in (let x66 = (one F./ (F.sqrt (x65 F.+ (one F./ fromi64 100000))))
    in (let x67 = (imap2 4 4 (\x69_0 x69_1 -> (x21[x64_0][x69_0][x69_1] F.* x66)))
    in x67[x64_1][x64_2])))))
    in (let x59 = (imap2 16 64 (\x70_0 x70_1 -> (isum2 4 4 (\x71_0 x71_1 -> (wup[x70_1][((x71_0 * 4) + x71_1)] F.* x58[x70_0][x71_0][x71_1])))))
    in (let x60 = (imap2 16 64 (\x72_0 x72_1 -> (if (zero F.<= x59[x72_0][x72_1]) then x59[x72_0][x72_1] else zero)))
    in (let x61 = (imap3 16 4 4 (\x73_0 x73_1 x73_2 -> (isum1 64 (\x74_0 -> (wdown[((x73_1 * 4) + x73_2)][x74_0] F.* x60[x73_0][x74_0])))))
    in (let x62 = (imap3 16 4 4 (\x75_0 x75_1 x75_2 -> (x61[x75_0][x75_1][x75_2] F.+ x21[x75_0][x75_1][x75_2])))
    in (imap3 16 4 4 (\x63_0 x63_1 x63_2 -> x62[x63_0][x63_1][x63_2])))))))
    in (imap3 16 4 4 (\x23_0 x23_1 x23_2 -> x22[x23_0][x23_1][x23_2]))))))))))
    in (let x6 = (imap2 16 27 (\x76_0 x76_1 -> (isum2 4 4 (\x77_0 x77_1 -> (wvoc[x76_1][((x77_0 * 4) + x77_1)] F.* x5[x76_0][x77_0][x77_1])))))
    in (imap2 16 27 (\x7_0 x7_1 -> x6[x7_0][x7_1]))))))
    in (let x1 = (imap1 16 (\x78_0 -> (let x79 = (let x82 = (imap1 27 (\x86_0 -> (F.exp x0[x78_0][x86_0])))
    in (let x83 = (isum1 27 (\x87_0 -> x82[x87_0]))
    in (let x84 = (imap1 27 (\x88_0 -> (x82[x88_0] F.* (one F./ x83))))
    in (imap1 27 (\x85_0 -> x84[x85_0])))))
    in (let x80 = (imap1 27 (\x89_0 -> (F.log x79[x89_0])))
    in (let x81 = (F.neg (isum1 27 (\x90_0 -> (x80[x90_0] F.* target[x78_0][x90_0]))))
    in x81)))))
    in (
      let x2 = ((isum1 16 (\x91_0 -> x1[x91_0])) F./ fromi64 16)
    -- in x2)))
    in
    let loss = x2
    let losses = x1
    in (loss, losses)
    )))

  -- is this correct? does it fill with zeroes if sl < 16?
  def cal_target [asl] : (target_ids : [asl]i64) -> [16][27]real =
    \(target_ids : [asl]i64) ->
    imap2 16 27 (\n m -> (if (n < asl && target_ids[n] == m) then one else zero))

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
    (wqry, wqry, wqry, wqry, wqry, wup, wdown, wvoc, wseq)


    -- in (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq)
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
--    in nn64.grad_loss mask wpe wqry wkey wval wout wup wdown wvoc wseq target

