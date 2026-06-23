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

def train_gen : (mask: [16][16]real)
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
    in (let x15 = (imap3 16 4 4 (\x31_0 x31_1 x31_2 -> (isum2 4 4 (\x32_0 x32_1 -> (wval[((x31_1 * 4) + x31_2)][((x32_0 * 4) + x32_1)] F.* x12[x31_0][x32_0][x32_1])))))
    in (let x16 = (imap3 16 4 4 (\x33_0 x33_1 x33_2 -> (let x34 = (imap2 16 16 (\x39_0 x39_1 -> (isum1 4 (\x40_0 -> (x13[x39_0][x33_1][x40_0] F.* x14[x39_1][x33_1][x40_0])))))
    in (let x35 = (imap2 16 16 (\x41_0 x41_1 -> (x34[x41_0][x41_1] F./ fromi64 2)))
    in (let x36 = (imap2 16 16 (\x42_0 x42_1 -> (x35[x42_0][x42_1] F.+ mask[x42_0][x42_1])))
    in (let x37 = (imap2 16 16 (\x43_0 x43_1 -> (let x44 = (imap1 16 (\x47_0 -> (F.exp x36[x43_0][x47_0])))
    in (let x45 = (isum1 16 (\x48_0 -> x44[x48_0]))
    in (let x46 = (imap1 16 (\x49_0 -> (x44[x49_0] F.* (one F./ x45))))
    in x46[x43_1])))))
    in (let x38 = (imap2 16 4 (\x50_0 x50_1 -> (isum1 16 (\x51_0 -> (x37[x50_0][x51_0] F.* x15[x51_0][x33_1][x50_1])))))
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

  -- def train_gen : (mask: [16][16]real)
  --   -> (wpe: [16][16]real)
  --   -> (wqry: [16][16]real)
  --   -> (wkey: [16][16]real)
  --   -> (wval: [16][16]real)
  --   -> (wout: [16][16]real)
  --   -> (wup: [64][16]real)
  --   -> (wdown: [16][64]real)
  --   -> (wvoc: [27][16]real)
  --   -> (wseq: [16][27]real)
  --   -> (target: [16][27]real)
  --   -> (
  --      [16][16]real
  --      , -- dwpe
  --      [16][16]real
  --      , -- dwqry
  --      [16][16]real
  --      , -- dwkey
  --      [16][16]real
  --      , -- dwval
  --      [16][16]real
  --      , -- dwout
  --      [64][16]real
  --      , -- dwup
  --      [16][64]real
  --      , -- dwdown
  --      [27][16]real
  --      , -- dwvoc
  --      [16][16]real
  --      , -- dwseq
  --      real
  --      -- loss
  --      ) =
  --   #[unsafe]
  --   \(mask: [16][16]real) (wpe: [16][16]real) (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real) (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real) (wvoc: [27][16]real) (wseq: [16][27]real) (target: [16][27]real) ->

  --   let x0 = 
    
  --   let loss = zero 
  --   in (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq, loss)
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

entry make_params (wte: [27][16]f64)  (wpe: [16][16]f64) 
    (wqry: [16][16]f64) (wkey: [16][16]f64) (wval: [16][16]f64) 
    (wout: [16][16]f64) (wup: [64][16]f64) (wdown: [16][64]f64) 
    (wvoc: [27][16]f64) : params = 
    {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc}

def main (p : params) (tok_ids : [16]i64) (mask : [16][16]f64) : [16][27]f64 = 
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let wseq = (imap2 16 16 (\m n -> wte[tok_ids[m]][n]))
   in nn64.train_gen mask wpe wqry wkey wval wout wup wdown wvoc wseq

-- type~ str_pair = ([]u8, []u8) --??

-- entry convert (imgs_bytes: []u8) (lbls_bytes: []u8) : str_pair =
--   (imgs_bytes, lbls_bytes)

-- type state =
--   { k1: [6][5][5]f64
--   , b1: [6]f64
--   , k2: [12][6][5][5]f64
--   , b2: [12]f64
--   , fc: [10][12][4][4]f64
--   , b: [10]f64
--   }

-- entry iteration [n] (trainings: i64) (batchsize: i64) (rate: f64) (imgs: [n][28][28]f64) (lbls: [n]i8) (s: state) : (state, f64) =
--   let gen_target i = imap 10 (\j -> if j == i then 1.0 else 0.0)
--   let avg (a: []f64) = nn64.sum a / f64.i64 (length a)
--   let (s, err) =
--     loop (s, err) = (s, 0.0)
--     for i < trainings / batchsize do
--       let {k1, b1, k2, b2, fc, b} = s
--       -- This is where we call trainings in parallel!
--       let r =
--         imap batchsize (\j ->
--                           let img = imgs[i * batchsize + j]
--                           let lbl = gen_target (i64.i8 lbls[i * batchsize + j])
--                           in nn64.train_gen img k1 b1 k2 b2 fc b lbl)
--       let (bdk1, bdb1, bdk2, bdb2, bdfc, bdb, berr) = unzip7 r
--       -- TODO: these should happen in-place, but hopefully this is not
--       --       a hotspot, the arrays are rather small.
--       let k1' =
--         imap3 6 5 5 (\i j k ->
--                        k1[i][j][k] - rate * (avg (imap batchsize (\t -> bdk1[t][i][j][k]))))
--       let b1' =
--         imap1 6 (\i ->
--                    b1[i] - rate * (avg (imap batchsize (\t -> bdb1[t][i]))))
--       let k2' =
--         imap4 12 6 5 5 (\i j k l ->
--                           k2[i][j][k][l] - rate * (avg (imap batchsize (\t -> bdk2[t][i][j][k][l]))))
--       let b2' =
--         imap1 12 (\i ->
--                     b2[i] - rate * (avg (imap batchsize (\t -> bdb2[t][i]))))
--       let fc' =
--         imap4 10 12 4 4 (\i j k l ->
--                            fc[i][j][k][l] - rate * (avg (imap batchsize (\t -> bdfc[t][i][j][k][l]))))
--       let b' =
--         imap1 10 (\i ->
--                     b[i] - rate * (avg (imap batchsize (\t -> bdb[t][i]))))
--       let err' = err + nn64.sum berr
--       in ( {k1 = k1', b1 = b1', k2 = k2', b2 = b2', fc = fc', b = b'}
--          , err'
--          )
--   in (s, err / 10.0 / f64.i64 trainings)

-- entry initial_state : state =
--   let k1 = imap3 6 5 5 (\_ _ _ -> 1.0 / 25.0)
--   let b1 = imap1 6 (\_ -> 1.0 / 6.0)
--   let k2 = imap4 12 6 5 5 (\_ _ _ _ -> 1.0 / 150.0)
--   let b2 = imap1 12 (\_ -> 1.0 / 12.0)
--   let fc = imap4 10 12 4 4 (\_ _ _ _ -> 1.0 / 192.0)
--   let b = imap1 10 (\_ -> 1.0 / 10.0)
--   in {k1, b1, k2, b2, fc, b}
