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

  def SL : i64 = 16
  def VO : i64 = 27
  def ED : i64 = 16
  def FD : i64 = 64

  --==== This is the generated function. ====--
  def train_gen : (mask: [SL][SL]real)
    -> (wpe: [SL][ED]real)
    -> (wqry: [ED][ED]real)
    -> (wkey: [ED][ED]real)
    -> (wval: [ED][ED]real)
    -> (wout: [ED][ED]real)
    -> (wup: [FD][ED]real)
    -> (wdown: [ED][FD]real)
    -> (wvoc: [VO][ED]real)
    -> (wseq: [SL][VO]real)
    -> (target: [SL][VO]real)
    -> (
       [SL][ED]real
       , -- dwpe
       [ED][ED]real
       , -- dwqry
       [ED][ED]real
       , -- dwkey
       [ED][ED]real
       , -- dwval
       [ED][ED]real
       , -- dwout
       [FD][ED]real
       , -- dwup
       [ED][FD]real
       , -- dwdown
       [VO][ED]real
       , -- dwvoc
       [SL][VO]real
       -- swseq
       ) =
    --#[unsafe]
    \(mask: [SL][SL]real) (wpe: [SL][ED]real) (wqry: [ED][ED]real)
    (wkey: [ED][ED]real) (wval: [ED][ED]real) (wout: [ED][ED]real)
    (wup: [FD][ED]real) (wdown: [ED][FD]real)
    (wvoc: [VO][ED]real) (wseq: [SL][VO]real) (target: [SL][VO]real) ->

    let x0 = (let x1 = (imap2 16 16 (\ x6_0 x6_1 -> (wpe[x6_0][x6_1] F.+ wseq[x6_0][x6_1])))
    in (let x2 = (imap3 16 4 4 (\ x7_0 x7_1 x7_2 -> x1[x7_0][((x7_1 * 4) + x7_2)]))
    in (let x3 = (let x8 = (imap3 16 4 4 (\ x17_0 x17_1 x17_2 -> (let x18 = ((isum2 4 4 (\ x21_0 x21_1 -> (x2[x17_0][x21_0][x21_1] F.* x2[x17_0][x21_0][x21_1]))) F./ fromi64 16)
    in (let x19 = (one F./ (F.sqrt x18))
    in (let x20 = (imap2 4 4 (\ x22_0 x22_1 -> (x2[x17_0][x22_0][x22_1] F.* x19)))
    in x20[x17_1][x17_2])))))
    in (let x9 = (imap3 16 4 4 (\ x23_0 x23_1 x23_2 -> (isum2 4 4 (\ x24_0 x24_1 -> (wqry[((x23_1 * 4) + x23_2)][((x24_0 * 4) + x24_1)] F.* x8[x23_0][x24_0][x24_1])))))
    in (let x10 = (imap3 16 4 4 (\ x25_0 x25_1 x25_2 -> (isum2 4 4 (\ x26_0 x26_1 -> (wkey[((x25_1 * 4) + x25_2)][((x26_0 * 4) + x26_1)] F.* x8[x25_0][x26_0][x26_1])))))
    in (let x11 = (imap3 16 4 4 (\ x27_0 x27_1 x27_2 -> (isum2 4 4 (\ x28_0 x28_1 -> (wval[((x27_1 * 4) + x27_2)][((x28_0 * 4) + x28_1)] F.* x8[x27_0][x28_0][x28_1])))))
    in (let x12 = (imap3 16 4 4 (\ x29_0 x29_1 x29_2 -> (isum1 16 (\ x30_0 -> (((F.exp (((isum1 4 (\ x32_0 -> (x9[x29_0][x29_1][x32_0] F.* x10[x30_0][x29_1][x32_0]))) F./ fromi64 2) F.+ mask[x29_0][x30_0])) F.* (one F./ (isum1 16 (\ x31_0 -> (F.exp (((isum1 4 (\ x33_0 -> (x9[x29_0][x29_1][x33_0] F.* x10[x31_0][x29_1][x33_0]))) F./ fromi64 2) F.+ mask[x29_0][x31_0])))))) F.* x11[x30_0][x29_1][x29_2])))))
    in (let x13 = (imap3 16 4 4 (\ x34_0 x34_1 x34_2 -> (isum2 4 4 (\ x35_0 x35_1 -> (wout[((x34_1 * 4) + x34_2)][((x35_0 * 4) + x35_1)] F.* x12[x34_0][x35_0][x35_1])))))
    in (let x14 = (imap3 16 4 4 (\ x36_0 x36_1 x36_2 -> (x13[x36_0][x36_1][x36_2] F.+ x2[x36_0][x36_1][x36_2])))
    in (let x15 = (let x37 = (imap3 16 4 4 (\ x43_0 x43_1 x43_2 -> (let x44 = ((isum2 4 4 (\ x47_0 x47_1 -> (x14[x43_0][x47_0][x47_1] F.* x14[x43_0][x47_0][x47_1]))) F./ fromi64 16)
    in (let x45 = (one F./ (F.sqrt x44))
    in (let x46 = (imap2 4 4 (\ x48_0 x48_1 -> (x14[x43_0][x48_0][x48_1] F.* x45)))
    in x46[x43_1][x43_2]))))) 
    -- in (let x38 = (imap3 16 64 16 (\ x49_0 x49_1 x49_2 -> (isum2 4 4 (\ x50_0 x50_1 -> (wup[x49_1][x49_2][((x50_0 * 4) + x50_1)] F.* x37[x49_0][x50_0][x50_1])))))
    in (let x39 = (imap3 16 64 16 (\ x51_0 x51_1 x51_2 -> (if (zero <= x38[x51_0][x51_1][x51_2]) then x38[x51_0][x51_1][x51_2] else zero)))
    in (let x40 = (imap3 16 4 4 (\ x52_0 x52_1 x52_2 -> (isum2 64 16 (\ x53_0 x53_1 -> (wdown[((x52_1 * 4) + x52_2)][x53_0][x53_1] F.* x39[x52_0][x53_0][x53_1])))))
    in (let x41 = (imap3 16 4 4 (\ x54_0 x54_1 x54_2 -> (x40[x54_0][x54_1][x54_2] F.+ x14[x54_0][x54_1][x54_2])))
    in (imap3 16 4 4 (\ x42_0 x42_1 x42_2 -> x41[x42_0][x42_1][x42_2])))))))
    in (imap3 16 4 4 (\ x16_0 x16_1 x16_2 -> x15[x16_0][x16_1][x16_2]))))))))))
    in (let x4 = (imap2 16 27 (\ x55_0 x55_1 -> (isum2 4 4 (\ x56_0 x56_1 -> (wvoc[x55_1][((x56_0 * 4) + x56_1)] F.* x3[x55_0][x56_0][x56_1])))))
    in (imap2 16 27 (\ x5_0 x5_1 -> x4[x5_0][x5_1]))))))
    let x57 = (imap1 16 (\ x58_0 -> (F.neg (isum1 27 (\ x59_0 -> (target[x58_0][x59_0] F.* (F.log ((F.exp x0[x58_0][x59_0]) F.* (one F./ (F.exp x0[x58_0][x59_0]))))))))))
    let x61 = ((isum1 16 (\ x62_0 -> x57[x62_0])) F./ fromi64 16)
    let x63 = one
    let x64 = (imap1 16 (\ x65_0 -> (x63 F./ fromi64 16)))
    let x66 = (imap2 16 27 (\ x70_0 x70_1 -> (isum1 16 (\ x67_0 -> (isum1 27 (\ x68_0 -> ((if ((x70_0 == x67_0)) then (if ((x70_1 == x68_0)) then ((F.exp x0[x67_0][x68_0]) F.* (F.neg (((((F.neg x64[x67_0]) F.* target[x67_0][x68_0]) F.* (one F./ ((F.exp x0[x67_0][x68_0]) F.* (one F./ (F.exp x0[x67_0][x68_0]))))) F.* (F.exp x0[x67_0][x68_0])) F.* (one F./ ((F.exp x0[x67_0][x68_0]) F.* (F.exp x0[x67_0][x68_0])))))) else zero) else zero) F.+ (if ((x70_0 == x67_0)) then (if ((x70_1 == x68_0)) then (if () then ((F.exp x0[x67_0][x68_0]) F.* ((((F.neg x64[x67_0]) F.* target[x67_0][x68_0]) F.* (one F./ ((F.exp x0[x67_0][x68_0]) F.* (one F./ (F.exp x0[x67_0][x68_0]))))) F.* (one F./ (F.exp x0[x67_0][x68_0])))) else zero) else zero) else zero))))))))
    let x76 = (imap2 16 16 (\ x77_0 x77_1 -> (wpe[x77_0][x77_1] F.+ wseq[x77_0][x77_1])))
    let x78 = (imap3 16 4 4 (\ x79_0 x79_1 x79_2 -> x76[x79_0][((x79_1 * 4) + x79_2)]))
    let x80 = (let x81 = (imap3 16 4 4 (\ x90_0 x90_1 x90_2 -> (let x91 = ((isum2 4 4 (\ x94_0 x94_1 -> (x78[x90_0][x94_0][x94_1] F.* x78[x90_0][x94_0][x94_1]))) F./ fromi64 16)
    in (let x92 = (one F./ (F.sqrt x91))
    in (let x93 = (imap2 4 4 (\ x95_0 x95_1 -> (x78[x90_0][x95_0][x95_1] F.* x92)))
    in x93[x90_1][x90_2])))))
    in (let x82 = (imap3 16 4 4 (\ x96_0 x96_1 x96_2 -> (isum2 4 4 (\ x97_0 x97_1 -> (wqry[((x96_1 * 4) + x96_2)][((x97_0 * 4) + x97_1)] F.* x81[x96_0][x97_0][x97_1])))))
    in (let x83 = (imap3 16 4 4 (\ x98_0 x98_1 x98_2 -> (isum2 4 4 (\ x99_0 x99_1 -> (wkey[((x98_1 * 4) + x98_2)][((x99_0 * 4) + x99_1)] F.* x81[x98_0][x99_0][x99_1])))))
    in (let x84 = (imap3 16 4 4 (\ x100_0 x100_1 x100_2 -> (isum2 4 4 (\ x101_0 x101_1 -> (wval[((x100_1 * 4) + x100_2)][((x101_0 * 4) + x101_1)] F.* x81[x100_0][x101_0][x101_1])))))
    in (let x85 = (imap3 16 4 4 (\ x102_0 x102_1 x102_2 -> (isum1 16 (\ x103_0 -> (((F.exp (((isum1 4 (\ x105_0 -> (x82[x102_0][x102_1][x105_0] F.* x83[x103_0][x102_1][x105_0]))) F./ fromi64 2) F.+ mask[x102_0][x103_0])) F.* (one F./ (isum1 16 (\ x104_0 -> (F.exp (((isum1 4 (\ x106_0 -> (x82[x102_0][x102_1][x106_0] F.* x83[x104_0][x102_1][x106_0]))) F./ fromi64 2) F.+ mask[x102_0][x104_0])))))) F.* x84[x103_0][x102_1][x102_2])))))
    in (let x86 = (imap3 16 4 4 (\ x107_0 x107_1 x107_2 -> (isum2 4 4 (\ x108_0 x108_1 -> (wout[((x107_1 * 4) + x107_2)][((x108_0 * 4) + x108_1)] F.* x85[x107_0][x108_0][x108_1])))))
    in (let x87 = (imap3 16 4 4 (\ x109_0 x109_1 x109_2 -> (x86[x109_0][x109_1][x109_2] F.+ x78[x109_0][x109_1][x109_2])))
    in (let x88 = (let x110 = (imap3 16 4 4 (\ x116_0 x116_1 x116_2 -> (let x117 = ((isum2 4 4 (\ x120_0 x120_1 -> (x87[x116_0][x120_0][x120_1] F.* x87[x116_0][x120_0][x120_1]))) F./ fromi64 16)
    in (let x118 = (one F./ (F.sqrt x117))
    in (let x119 = (imap2 4 4 (\ x121_0 x121_1 -> (x87[x116_0][x121_0][x121_1] F.* x118)))
    in x119[x116_1][x116_2])))))
    in (let x111 = (imap3 16 64 16 (\ x122_0 x122_1 x122_2 -> (isum2 4 4 (\ x123_0 x123_1 -> (wup[x122_1][x122_2][((x123_0 * 4) + x123_1)] F.* x110[x122_0][x123_0][x123_1])))))
    in (let x112 = (imap3 16 64 16 (\ x124_0 x124_1 x124_2 -> (if (zero <= x111[x124_0][x124_1][x124_2]) then x111[x124_0][x124_1][x124_2] else zero)))
    in (let x113 = (imap3 16 4 4 (\ x125_0 x125_1 x125_2 -> (isum2 64 16 (\ x126_0 x126_1 -> (wdown[((x125_1 * 4) + x125_2)][x126_0][x126_1] F.* x112[x125_0][x126_0][x126_1])))))
    in (let x114 = (imap3 16 4 4 (\ x127_0 x127_1 x127_2 -> (x113[x127_0][x127_1][x127_2] F.+ x87[x127_0][x127_1][x127_2])))
    in (imap3 16 4 4 (\ x115_0 x115_1 x115_2 -> x114[x115_0][x115_1][x115_2])))))))
    in (imap3 16 4 4 (\ x89_0 x89_1 x89_2 -> x88[x89_0][x89_1][x89_2]))))))))))
    let x128 = (imap2 16 27 (\ x129_0 x129_1 -> (isum2 4 4 (\ x130_0 x130_1 -> (wvoc[x129_1][((x130_0 * 4) + x130_1)] F.* x80[x129_0][x130_0][x130_1])))))
    let x131 = (imap2 16 27 (\ x132_0 x132_1 -> x66[x132_0][x132_1]))
    let x133 = (imap3 16 4 4 (\ x134_0 x134_1 x134_2 -> (isum1 27 (\ x135_0 -> (x131[x134_0][x135_0] F.* wvoc[x135_0][((x134_1 * 4) + x134_2)])))))
    let x136 = (imap3 16 4 4 (\ x137_0 x137_1 x137_2 -> (let x138 = ((isum2 4 4 (\ x141_0 x141_1 -> (x78[x137_0][x141_0][x141_1] F.* x78[x137_0][x141_0][x141_1]))) F./ fromi64 16)
    in (let x139 = (one F./ (F.sqrt x138))
    in (let x140 = (imap2 4 4 (\ x142_0 x142_1 -> (x78[x137_0][x142_0][x142_1] F.* x139)))
    in x140[x137_1][x137_2])))))
    let x143 = (imap3 16 4 4 (\ x144_0 x144_1 x144_2 -> (isum2 4 4 (\ x145_0 x145_1 -> (wqry[((x144_1 * 4) + x144_2)][((x145_0 * 4) + x145_1)] F.* x136[x144_0][x145_0][x145_1])))))
    let x146 = (imap3 16 4 4 (\ x147_0 x147_1 x147_2 -> (isum2 4 4 (\ x148_0 x148_1 -> (wkey[((x147_1 * 4) + x147_2)][((x148_0 * 4) + x148_1)] F.* x136[x147_0][x148_0][x148_1])))))
    let x149 = (imap3 16 4 4 (\ x150_0 x150_1 x150_2 -> (isum2 4 4 (\ x151_0 x151_1 -> (wval[((x150_1 * 4) + x150_2)][((x151_0 * 4) + x151_1)] F.* x136[x150_0][x151_0][x151_1])))))
    let x152 = (imap3 16 4 4 (\ x153_0 x153_1 x153_2 -> (isum1 16 (\ x154_0 -> (((F.exp (((isum1 4 (\ x156_0 -> (x143[x153_0][x153_1][x156_0] F.* x146[x154_0][x153_1][x156_0]))) F./ fromi64 2) F.+ mask[x153_0][x154_0])) F.* (one F./ (isum1 16 (\ x155_0 -> (F.exp (((isum1 4 (\ x157_0 -> (x143[x153_0][x153_1][x157_0] F.* x146[x155_0][x153_1][x157_0]))) F./ fromi64 2) F.+ mask[x153_0][x155_0])))))) F.* x149[x154_0][x153_1][x153_2])))))
    let x158 = (imap3 16 4 4 (\ x159_0 x159_1 x159_2 -> (isum2 4 4 (\ x160_0 x160_1 -> (wout[((x159_1 * 4) + x159_2)][((x160_0 * 4) + x160_1)] F.* x152[x159_0][x160_0][x160_1])))))
    let x161 = (imap3 16 4 4 (\ x162_0 x162_1 x162_2 -> (x158[x162_0][x162_1][x162_2] F.+ x78[x162_0][x162_1][x162_2])))
    let x163 = (let x164 = (imap3 16 4 4 (\ x170_0 x170_1 x170_2 -> (let x171 = ((isum2 4 4 (\ x174_0 x174_1 -> (x161[x170_0][x174_0][x174_1] F.* x161[x170_0][x174_0][x174_1]))) F./ fromi64 16)
    in (let x172 = (one F./ (F.sqrt x171))
    in (let x173 = (imap2 4 4 (\ x175_0 x175_1 -> (x161[x170_0][x175_0][x175_1] F.* x172)))
    in x173[x170_1][x170_2])))))
    in (let x165 = (imap3 16 64 16 (\ x176_0 x176_1 x176_2 -> (isum2 4 4 (\ x177_0 x177_1 -> (wup[x176_1][x176_2][((x177_0 * 4) + x177_1)] F.* x164[x176_0][x177_0][x177_1])))))
    in (let x166 = (imap3 16 64 16 (\ x178_0 x178_1 x178_2 -> (if (zero <= x165[x178_0][x178_1][x178_2]) then x165[x178_0][x178_1][x178_2] else zero)))
    in (let x167 = (imap3 16 4 4 (\ x179_0 x179_1 x179_2 -> (isum2 64 16 (\ x180_0 x180_1 -> (wdown[((x179_1 * 4) + x179_2)][x180_0][x180_1] F.* x166[x179_0][x180_0][x180_1])))))
    in (let x168 = (imap3 16 4 4 (\ x181_0 x181_1 x181_2 -> (x167[x181_0][x181_1][x181_2] F.+ x161[x181_0][x181_1][x181_2])))
    in (imap3 16 4 4 (\ x169_0 x169_1 x169_2 -> x168[x169_0][x169_1][x169_2])))))))
    let x182 = (imap3 16 4 4 (\ x183_0 x183_1 x183_2 -> x133[x183_0][x183_1][x183_2]))
    let x184 = (imap3 16 4 4 (\ x185_0 x185_1 x185_2 -> (let x186 = ((isum2 4 4 (\ x189_0 x189_1 -> (x161[x185_0][x189_0][x189_1] F.* x161[x185_0][x189_0][x189_1]))) F./ fromi64 16)
    in (let x187 = (one F./ (F.sqrt x186))
    in (let x188 = (imap2 4 4 (\ x190_0 x190_1 -> (x161[x185_0][x190_0][x190_1] F.* x187)))
    in x188[x185_1][x185_2])))))
    let x191 = (imap3 16 64 16 (\ x192_0 x192_1 x192_2 -> (isum2 4 4 (\ x193_0 x193_1 -> (wup[x192_1][x192_2][((x193_0 * 4) + x193_1)] F.* x184[x192_0][x193_0][x193_1])))))
    let x194 = (imap3 16 64 16 (\ x195_0 x195_1 x195_2 -> (if (zero <= x191[x195_0][x195_1][x195_2]) then x191[x195_0][x195_1][x195_2] else zero)))
    let x196 = (imap3 16 4 4 (\ x197_0 x197_1 x197_2 -> (isum2 64 16 (\ x198_0 x198_1 -> (wdown[((x197_1 * 4) + x197_2)][x198_0][x198_1] F.* x194[x197_0][x198_0][x198_1])))))
    let x199 = (imap3 16 4 4 (\ x200_0 x200_1 x200_2 -> (x196[x200_0][x200_1][x200_2] F.+ x161[x200_0][x200_1][x200_2])))
    let x201 = (imap3 16 4 4 (\ x202_0 x202_1 x202_2 -> x182[x202_0][x202_1][x202_2]))
    let x203 = (imap3 16 4 4 (\ x204_0 x204_1 x204_2 -> x201[x204_0][x204_1][x204_2]))
    let x205 = (imap3 16 64 16 (\ x206_0 x206_1 x206_2 -> (isum2 4 4 (\ x207_0 x207_1 -> (x203[x206_0][x207_0][x207_1] F.* wdown[((x207_0 * 4) + x207_1)][x206_1][x206_2])))))
    let x208 = (imap3 16 64 16 (\ x209_0 x209_1 x209_2 -> ((if (zero <= x191[x209_0][x209_1][x209_2]) then one else zero) F.* x205[x209_0][x209_1][x209_2])))
    let x210 = (imap3 16 4 4 (\ x211_0 x211_1 x211_2 -> (isum2 64 16 (\ x212_0 x212_1 -> (x208[x211_0][x212_0][x212_1] F.* wup[x212_0][x212_1][((x211_1 * 4) + x211_2)])))))
    let x213 = (imap3 16 4 4 (\ x222_0 x222_1 x222_2 -> (x203[x222_0][x222_1][x222_2] F.+ (isum1 16 (\ x214_0 -> (let x215 = ((isum2 4 4 (\ x223_0 x223_1 -> (x161[x214_0][x223_0][x223_1] F.* x161[x214_0][x223_0][x223_1]))) F./ fromi64 16)
    in (let x216 = (one F./ (F.sqrt x215))
    in (let x217 = (imap2 4 4 (\ x224_0 x224_1 -> (x161[x214_0][x224_0][x224_1] F.* x216)))
    in (let x218 = (imap2 4 4 (\ x225_0 x225_1 -> x210[x214_0][x225_0][x225_1]))
    in (let x219 = (isum2 4 4 (\ x226_0 x226_1 -> (x218[x226_0][x226_1] F.* x161[x214_0][x226_0][x226_1])))
    in (let x220 = ((F.neg (x219 F.* (one F./ ((F.sqrt x215) F.* (F.sqrt x215))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x215))))
    in ((if ((x222_0 == x214_0)) then (x218[x222_1][x222_2] F.* x216) else zero) F.+ (isum2 4 4 (\ x221_0 x221_1 -> ((if ((x222_0 == x214_0)) then (if ((x222_1 == x221_0) && (x222_2 == x221_1)) then ((x220 F./ fromi64 16) F.* x161[x214_0][x221_0][x221_1]) else zero) else zero) F.+ (if ((x222_0 == x214_0)) then (if ((x222_1 == x221_0) && (x222_2 == x221_1)) then ((x220 F./ fromi64 16) F.* x161[x214_0][x221_0][x221_1]) else zero) else zero))))))))))))))))
    let x227 = (imap3 16 4 4 (\ x235_0 x235_1 x235_2 -> ((isum1 16 (\ x228_0 -> (let x229 = ((isum2 4 4 (\ x236_0 x236_1 -> (x161[x228_0][x236_0][x236_1] F.* x161[x228_0][x236_0][x236_1]))) F./ fromi64 16)
    in (let x230 = (one F./ (F.sqrt x229))
    in (let x231 = (imap2 4 4 (\ x237_0 x237_1 -> (x161[x228_0][x237_0][x237_1] F.* x230)))
    in (let x232 = (imap2 4 4 (\ x238_0 x238_1 -> x210[x228_0][x238_0][x238_1]))
    in (let x233 = (isum2 4 4 (\ x239_0 x239_1 -> (x232[x239_0][x239_1] F.* x161[x228_0][x239_0][x239_1])))
    in (let x234 = ((F.neg (x233 F.* (one F./ ((F.sqrt x229) F.* (F.sqrt x229))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x229))))
    in zero)))))))) F.+ x213[x235_0][x235_1][x235_2])))
    let x240 = (imap3 16 4 4 (\ x248_0 x248_1 x248_2 -> ((isum1 16 (\ x241_0 -> (let x242 = ((isum2 4 4 (\ x249_0 x249_1 -> (x161[x241_0][x249_0][x249_1] F.* x161[x241_0][x249_0][x249_1]))) F./ fromi64 16)
    in (let x243 = (one F./ (F.sqrt x242))
    in (let x244 = (imap2 4 4 (\ x250_0 x250_1 -> (x161[x241_0][x250_0][x250_1] F.* x243)))
    in (let x245 = (imap2 4 4 (\ x251_0 x251_1 -> x210[x241_0][x251_0][x251_1]))
    in (let x246 = (isum2 4 4 (\ x252_0 x252_1 -> (x245[x252_0][x252_1] F.* x161[x241_0][x252_0][x252_1])))
    in (let x247 = ((F.neg (x246 F.* (one F./ ((F.sqrt x242) F.* (F.sqrt x242))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x242))))
    in zero)))))))) F.+ (isum2 4 4 (\ x253_0 x253_1 -> (x227[x248_0][x253_0][x253_1] F.* wout[((x253_0 * 4) + x253_1)][((x248_1 * 4) + x248_2)]))))))
    let x254 = (imap3 16 4 4 (\ x262_0 x262_1 x262_2 -> ((isum1 16 (\ x255_0 -> (let x256 = ((isum2 4 4 (\ x263_0 x263_1 -> (x161[x255_0][x263_0][x263_1] F.* x161[x255_0][x263_0][x263_1]))) F./ fromi64 16)
    in (let x257 = (one F./ (F.sqrt x256))
    in (let x258 = (imap2 4 4 (\ x264_0 x264_1 -> (x161[x255_0][x264_0][x264_1] F.* x257)))
    in (let x259 = (imap2 4 4 (\ x265_0 x265_1 -> x210[x255_0][x265_0][x265_1]))
    in (let x260 = (isum2 4 4 (\ x266_0 x266_1 -> (x259[x266_0][x266_1] F.* x161[x255_0][x266_0][x266_1])))
    in (let x261 = ((F.neg (x260 F.* (one F./ ((F.sqrt x256) F.* (F.sqrt x256))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x256))))
    in zero)))))))) F.+ (isum1 16 (\ x267_0 -> (isum1 4 (\ x268_0 -> (isum1 4 (\ x269_0 -> (isum2 16 4 (\ x270_0 x270_1 -> (isum1 4 (\ x271_0 -> (isum1 4 (\ x272_0 -> (isum1 16 (\ x273_0 -> (isum1 4 (\ x274_0 -> (isum1 16 (\ x275_0 -> (isum1 16 (\ x276_0 -> (isum1 16 (\ x277_0 -> (isum1 4 (\ x278_0 -> (isum1 16 (\ x279_0 -> (isum1 4 (\ x280_0 -> (if ((x262_1 == x278_0)) then (if ((x262_2 == x280_0)) then (if ((x262_0 == x279_0)) then (if ((x278_0 == x272_0)) then (if ((x279_0 == x277_0)) then (if ((x280_0 == x274_0)) then (if ((x277_0 == x276_0)) then ((if ((x275_0 == x273_0)) then (if ((x272_0 == x271_0)) then (if ((x273_0 == x270_0) && (x274_0 == x270_1)) then (if ((x270_0 == x267_0)) then (if ((x270_1 == x269_0)) then (if ((x271_0 == x268_0)) then x240[x267_0][x268_0][x269_0] else zero) else zero) else zero) else zero) else zero) else zero) F.* ((F.exp (((isum1 4 (\ x282_0 -> (x143[x275_0][x272_0][x282_0] F.* x146[x276_0][x272_0][x282_0]))) F./ fromi64 2) F.+ mask[x275_0][x276_0])) F.* (one F./ (isum1 16 (\ x281_0 -> (F.exp (((isum1 4 (\ x283_0 -> (x143[x275_0][x272_0][x283_0] F.* x146[x281_0][x272_0][x283_0]))) F./ fromi64 2) F.+ mask[x275_0][x281_0]))))))) else zero) else zero) else zero) else zero) else zero) else zero) else zero))))))))))))))))))))))))))))))))
    let x284 = (imap3 16 4 4 (\ x304_0 x304_1 x304_2 -> ((isum1 16 (\ x285_0 -> (let x286 = ((isum2 4 4 (\ x305_0 x305_1 -> (x161[x285_0][x305_0][x305_1] F.* x161[x285_0][x305_0][x305_1]))) F./ fromi64 16)
    in (let x287 = (one F./ (F.sqrt x286))
    in (let x288 = (imap2 4 4 (\ x306_0 x306_1 -> (x161[x285_0][x306_0][x306_1] F.* x287)))
    in (let x289 = (imap2 4 4 (\ x307_0 x307_1 -> x210[x285_0][x307_0][x307_1]))
    in (let x290 = (isum2 4 4 (\ x308_0 x308_1 -> (x289[x308_0][x308_1] F.* x161[x285_0][x308_0][x308_1])))
    in (let x291 = ((F.neg (x290 F.* (one F./ ((F.sqrt x286) F.* (F.sqrt x286))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x286))))
    in zero)))))))) F.+ (isum1 16 (\ x292_0 -> (isum1 4 (\ x293_0 -> (isum1 4 (\ x294_0 -> (isum2 16 4 (\ x295_0 x295_1 -> (isum1 4 (\ x296_0 -> (isum1 4 (\ x297_0 -> (isum1 16 (\ x298_0 -> (isum1 4 (\ x299_0 -> (isum1 16 (\ x300_0 -> (isum1 16 (\ x301_0 -> (isum1 16 (\ x302_0 -> (isum1 16 (\ x303_0 -> ((isum1 16 (\ x309_0 -> (isum1 16 (\ x310_0 -> (isum1 16 (\ x311_0 -> (isum1 16 (\ x312_0 -> (isum1 4 (\ x313_0 -> (isum1 4 (\ x314_0 -> (isum1 4 (\ x315_0 -> (isum1 16 (\ x316_0 -> (isum1 4 (\ x317_0 -> (isum1 16 (\ x318_0 -> (isum1 4 (\ x319_0 -> (if ((x304_1 == x317_0)) then (if ((x304_2 == x319_0)) then (if ((x304_0 == x318_0)) then (if ((x317_0 == x297_0)) then (if ((x318_0 == x316_0)) then (if ((x319_0 == x315_0)) then (if ((x315_0 == x314_0)) then (if ((x316_0 == x311_0)) then (if ((x314_0 == x313_0)) then ((if ((x312_0 == x310_0)) then ((if ((x310_0 == x302_0)) then (if ((x311_0 == x309_0)) then ((F.exp (((isum1 4 (\ x322_0 -> (x143[x302_0][x297_0][x322_0] F.* x146[x309_0][x297_0][x322_0]))) F./ fromi64 2) F.+ mask[x302_0][x309_0])) F.* (F.neg (((if ((x302_0 == x300_0)) then (if ((x303_0 == x301_0)) then ((if ((x300_0 == x298_0)) then (if ((x297_0 == x296_0)) then (if ((x298_0 == x295_0) && (x299_0 == x295_1)) then (if ((x295_0 == x292_0)) then (if ((x295_1 == x294_0)) then (if ((x296_0 == x293_0)) then x240[x292_0][x293_0][x294_0] else zero) else zero) else zero) else zero) else zero) else zero) F.* x149[x301_0][x297_0][x299_0]) else zero) else zero) F.* (F.exp (((isum1 4 (\ x323_0 -> (x143[x302_0][x297_0][x323_0] F.* x146[x303_0][x297_0][x323_0]))) F./ fromi64 2) F.+ mask[x302_0][x303_0]))) F.* (one F./ ((isum1 16 (\ x320_0 -> (F.exp (((isum1 4 (\ x324_0 -> (x143[x302_0][x297_0][x324_0] F.* x146[x320_0][x297_0][x324_0]))) F./ fromi64 2) F.+ mask[x302_0][x320_0])))) F.* (isum1 16 (\ x321_0 -> (F.exp (((isum1 4 (\ x325_0 -> (x143[x302_0][x297_0][x325_0] F.* x146[x321_0][x297_0][x325_0]))) F./ fromi64 2) F.+ mask[x302_0][x321_0]))))))))) else zero) else zero) F./ fromi64 2) else zero) F.* x143[x312_0][x297_0][x313_0]) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero))))))))))))))))))))))) F.+ (isum1 16 (\ x326_0 -> (isum1 16 (\ x327_0 -> (isum1 16 (\ x328_0 -> (isum1 4 (\ x329_0 -> (isum1 4 (\ x330_0 -> (isum1 4 (\ x331_0 -> (isum1 16 (\ x332_0 -> (isum1 4 (\ x333_0 -> (isum1 16 (\ x334_0 -> (isum1 4 (\ x335_0 -> (if ((x304_1 == x333_0)) then (if ((x304_2 == x335_0)) then (if ((x304_0 == x334_0)) then (if ((x333_0 == x297_0)) then (if ((x334_0 == x332_0)) then (if ((x335_0 == x331_0)) then (if ((x331_0 == x330_0)) then (if ((x332_0 == x327_0)) then (if ((x330_0 == x329_0)) then ((if ((x328_0 == x326_0)) then ((if ((x326_0 == x302_0)) then (if ((x327_0 == x303_0)) then ((F.exp (((isum1 4 (\ x337_0 -> (x143[x302_0][x297_0][x337_0] F.* x146[x303_0][x297_0][x337_0]))) F./ fromi64 2) F.+ mask[x302_0][x303_0])) F.* ((if ((x302_0 == x300_0)) then (if ((x303_0 == x301_0)) then ((if ((x300_0 == x298_0)) then (if ((x297_0 == x296_0)) then (if ((x298_0 == x295_0) && (x299_0 == x295_1)) then (if ((x295_0 == x292_0)) then (if ((x295_1 == x294_0)) then (if ((x296_0 == x293_0)) then x240[x292_0][x293_0][x294_0] else zero) else zero) else zero) else zero) else zero) else zero) F.* x149[x301_0][x297_0][x299_0]) else zero) else zero) F.* (one F./ (isum1 16 (\ x336_0 -> (F.exp (((isum1 4 (\ x338_0 -> (x143[x302_0][x297_0][x338_0] F.* x146[x336_0][x297_0][x338_0]))) F./ fromi64 2) F.+ mask[x302_0][x336_0]))))))) else zero) else zero) F./ fromi64 2) else zero) F.* x143[x328_0][x297_0][x329_0]) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero)))))))))))))))))))))))))))))))))))))))))))))))))
    let x339 = (imap3 16 4 4 (\ x359_0 x359_1 x359_2 -> ((isum1 16 (\ x340_0 -> (let x341 = ((isum2 4 4 (\ x360_0 x360_1 -> (x161[x340_0][x360_0][x360_1] F.* x161[x340_0][x360_0][x360_1]))) F./ fromi64 16)
    in (let x342 = (one F./ (F.sqrt x341))
    in (let x343 = (imap2 4 4 (\ x361_0 x361_1 -> (x161[x340_0][x361_0][x361_1] F.* x342)))
    in (let x344 = (imap2 4 4 (\ x362_0 x362_1 -> x210[x340_0][x362_0][x362_1]))
    in (let x345 = (isum2 4 4 (\ x363_0 x363_1 -> (x344[x363_0][x363_1] F.* x161[x340_0][x363_0][x363_1])))
    in (let x346 = ((F.neg (x345 F.* (one F./ ((F.sqrt x341) F.* (F.sqrt x341))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x341))))
    in zero)))))))) F.+ (isum1 16 (\ x347_0 -> (isum1 4 (\ x348_0 -> (isum1 4 (\ x349_0 -> (isum2 16 4 (\ x350_0 x350_1 -> (isum1 4 (\ x351_0 -> (isum1 4 (\ x352_0 -> (isum1 16 (\ x353_0 -> (isum1 4 (\ x354_0 -> (isum1 16 (\ x355_0 -> (isum1 16 (\ x356_0 -> (isum1 16 (\ x357_0 -> (isum1 16 (\ x358_0 -> ((isum1 16 (\ x364_0 -> (isum1 16 (\ x365_0 -> (isum1 16 (\ x366_0 -> (isum1 16 (\ x367_0 -> (isum1 4 (\ x368_0 -> (isum1 4 (\ x369_0 -> (isum1 16 (\ x370_0 -> (isum1 4 (\ x371_0 -> (if ((x359_1 == x369_0)) then (if ((x359_2 == x371_0)) then (if ((x359_0 == x370_0)) then (if ((x369_0 == x352_0)) then (if ((x370_0 == x367_0)) then (if ((x371_0 == x368_0)) then ((if ((x367_0 == x365_0)) then ((if ((x365_0 == x357_0)) then (if ((x366_0 == x364_0)) then ((F.exp (((isum1 4 (\ x374_0 -> (x143[x357_0][x352_0][x374_0] F.* x146[x364_0][x352_0][x374_0]))) F./ fromi64 2) F.+ mask[x357_0][x364_0])) F.* (F.neg (((if ((x357_0 == x355_0)) then (if ((x358_0 == x356_0)) then ((if ((x355_0 == x353_0)) then (if ((x352_0 == x351_0)) then (if ((x353_0 == x350_0) && (x354_0 == x350_1)) then (if ((x350_0 == x347_0)) then (if ((x350_1 == x349_0)) then (if ((x351_0 == x348_0)) then x240[x347_0][x348_0][x349_0] else zero) else zero) else zero) else zero) else zero) else zero) F.* x149[x356_0][x352_0][x354_0]) else zero) else zero) F.* (F.exp (((isum1 4 (\ x375_0 -> (x143[x357_0][x352_0][x375_0] F.* x146[x358_0][x352_0][x375_0]))) F./ fromi64 2) F.+ mask[x357_0][x358_0]))) F.* (one F./ ((isum1 16 (\ x372_0 -> (F.exp (((isum1 4 (\ x376_0 -> (x143[x357_0][x352_0][x376_0] F.* x146[x372_0][x352_0][x376_0]))) F./ fromi64 2) F.+ mask[x357_0][x372_0])))) F.* (isum1 16 (\ x373_0 -> (F.exp (((isum1 4 (\ x377_0 -> (x143[x357_0][x352_0][x377_0] F.* x146[x373_0][x352_0][x377_0]))) F./ fromi64 2) F.+ mask[x357_0][x373_0]))))))))) else zero) else zero) F./ fromi64 2) else zero) F.* x146[x366_0][x352_0][x368_0]) else zero) else zero) else zero) else zero) else zero) else zero))))))))))))))))) F.+ (isum1 16 (\ x378_0 -> (isum1 16 (\ x379_0 -> (isum1 16 (\ x380_0 -> (isum1 4 (\ x381_0 -> (isum1 4 (\ x382_0 -> (isum1 16 (\ x383_0 -> (isum1 4 (\ x384_0 -> (if ((x359_1 == x382_0)) then (if ((x359_2 == x384_0)) then (if ((x359_0 == x383_0)) then (if ((x382_0 == x352_0)) then (if ((x383_0 == x380_0)) then (if ((x384_0 == x381_0)) then ((if ((x380_0 == x378_0)) then ((if ((x378_0 == x357_0)) then (if ((x379_0 == x358_0)) then ((F.exp (((isum1 4 (\ x386_0 -> (x143[x357_0][x352_0][x386_0] F.* x146[x358_0][x352_0][x386_0]))) F./ fromi64 2) F.+ mask[x357_0][x358_0])) F.* ((if ((x357_0 == x355_0)) then (if ((x358_0 == x356_0)) then ((if ((x355_0 == x353_0)) then (if ((x352_0 == x351_0)) then (if ((x353_0 == x350_0) && (x354_0 == x350_1)) then (if ((x350_0 == x347_0)) then (if ((x350_1 == x349_0)) then (if ((x351_0 == x348_0)) then x240[x347_0][x348_0][x349_0] else zero) else zero) else zero) else zero) else zero) else zero) F.* x149[x356_0][x352_0][x354_0]) else zero) else zero) F.* (one F./ (isum1 16 (\ x385_0 -> (F.exp (((isum1 4 (\ x387_0 -> (x143[x357_0][x352_0][x387_0] F.* x146[x385_0][x352_0][x387_0]))) F./ fromi64 2) F.+ mask[x357_0][x385_0]))))))) else zero) else zero) F./ fromi64 2) else zero) F.* x146[x379_0][x352_0][x381_0]) else zero) else zero) else zero) else zero) else zero) else zero)))))))))))))))))))))))))))))))))))))))))))
    let x388 = (imap3 16 4 4 (\ x396_0 x396_1 x396_2 -> ((((isum1 16 (\ x389_0 -> (let x390 = ((isum2 4 4 (\ x397_0 x397_1 -> (x161[x389_0][x397_0][x397_1] F.* x161[x389_0][x397_0][x397_1]))) F./ fromi64 16)
    in (let x391 = (one F./ (F.sqrt x390))
    in (let x392 = (imap2 4 4 (\ x398_0 x398_1 -> (x161[x389_0][x398_0][x398_1] F.* x391)))
    in (let x393 = (imap2 4 4 (\ x399_0 x399_1 -> x210[x389_0][x399_0][x399_1]))
    in (let x394 = (isum2 4 4 (\ x400_0 x400_1 -> (x393[x400_0][x400_1] F.* x161[x389_0][x400_0][x400_1])))
    in (let x395 = ((F.neg (x394 F.* (one F./ ((F.sqrt x390) F.* (F.sqrt x390))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x390))))
    in zero)))))))) F.+ (isum2 4 4 (\ x401_0 x401_1 -> (x254[x396_0][x401_0][x401_1] F.* wval[((x401_0 * 4) + x401_1)][((x396_1 * 4) + x396_2)])))) F.+ (isum2 4 4 (\ x402_0 x402_1 -> (x284[x396_0][x402_0][x402_1] F.* wkey[((x402_0 * 4) + x402_1)][((x396_1 * 4) + x396_2)])))) F.+ (isum2 4 4 (\ x403_0 x403_1 -> (x339[x396_0][x403_0][x403_1] F.* wqry[((x403_0 * 4) + x403_1)][((x396_1 * 4) + x396_2)]))))))
    let x404 = (imap3 16 4 4 (\ x413_0 x413_1 x413_2 -> (x227[x413_0][x413_1][x413_2] F.+ (isum1 16 (\ x405_0 -> (let x406 = ((isum2 4 4 (\ x414_0 x414_1 -> (x78[x405_0][x414_0][x414_1] F.* x78[x405_0][x414_0][x414_1]))) F./ fromi64 16)
    in (let x407 = (one F./ (F.sqrt x406))
    in (let x408 = (imap2 4 4 (\ x415_0 x415_1 -> (x78[x405_0][x415_0][x415_1] F.* x407)))
    in (let x409 = (imap2 4 4 (\ x416_0 x416_1 -> x388[x405_0][x416_0][x416_1]))
    in (let x410 = (isum2 4 4 (\ x417_0 x417_1 -> (x409[x417_0][x417_1] F.* x78[x405_0][x417_0][x417_1])))
    in (let x411 = ((F.neg (x410 F.* (one F./ ((F.sqrt x406) F.* (F.sqrt x406))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x406))))
    in ((if ((x413_0 == x405_0)) then (x409[x413_1][x413_2] F.* x407) else zero) F.+ (isum2 4 4 (\ x412_0 x412_1 -> ((if ((x413_0 == x405_0)) then (if ((x413_1 == x412_0) && (x413_2 == x412_1)) then ((x411 F./ fromi64 16) F.* x78[x405_0][x412_0][x412_1]) else zero) else zero) F.+ (if ((x413_0 == x405_0)) then (if ((x413_1 == x412_0) && (x413_2 == x412_1)) then ((x411 F./ fromi64 16) F.* x78[x405_0][x412_0][x412_1]) else zero) else zero))))))))))))))))
    let x418 = (imap2 16 16 (\ x433_0 x433_1 -> (((isum1 16 (\ x419_0 -> (let x420 = ((isum2 4 4 (\ x434_0 x434_1 -> (x161[x419_0][x434_0][x434_1] F.* x161[x419_0][x434_0][x434_1]))) F./ fromi64 16)
    in (let x421 = (one F./ (F.sqrt x420))
    in (let x422 = (imap2 4 4 (\ x435_0 x435_1 -> (x161[x419_0][x435_0][x435_1] F.* x421)))
    in (let x423 = (imap2 4 4 (\ x436_0 x436_1 -> x210[x419_0][x436_0][x436_1]))
    in (let x424 = (isum2 4 4 (\ x437_0 x437_1 -> (x423[x437_0][x437_1] F.* x161[x419_0][x437_0][x437_1])))
    in (let x425 = ((F.neg (x424 F.* (one F./ ((F.sqrt x420) F.* (F.sqrt x420))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x420))))
    in zero)))))))) F.+ (isum1 16 (\ x426_0 -> (let x427 = ((isum2 4 4 (\ x438_0 x438_1 -> (x78[x426_0][x438_0][x438_1] F.* x78[x426_0][x438_0][x438_1]))) F./ fromi64 16)
    in (let x428 = (one F./ (F.sqrt x427))
    in (let x429 = (imap2 4 4 (\ x439_0 x439_1 -> (x78[x426_0][x439_0][x439_1] F.* x428)))
    in (let x430 = (imap2 4 4 (\ x440_0 x440_1 -> x388[x426_0][x440_0][x440_1]))
    in (let x431 = (isum2 4 4 (\ x441_0 x441_1 -> (x430[x441_0][x441_1] F.* x78[x426_0][x441_0][x441_1])))
    in (let x432 = ((F.neg (x431 F.* (one F./ ((F.sqrt x427) F.* (F.sqrt x427))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x427))))
    in zero))))))))) F.+ x404[x433_0][(x433_1 / 4)][(x433_1 % 4)])))

    let dmask = (imap2 16 16 (\ x468_0 x468_1 -> (((isum1 16 (\ x442_0 -> (let x443 = ((isum2 4 4 (\ x469_0 x469_1 -> (x161[x442_0][x469_0][x469_1] F.* x161[x442_0][x469_0][x469_1]))) F./ fromi64 16)
    in (let x444 = (one F./ (F.sqrt x443))
    in (let x445 = (imap2 4 4 (\ x470_0 x470_1 -> (x161[x442_0][x470_0][x470_1] F.* x444)))
    in (let x446 = (imap2 4 4 (\ x471_0 x471_1 -> x210[x442_0][x471_0][x471_1]))
    in (let x447 = (isum2 4 4 (\ x472_0 x472_1 -> (x446[x472_0][x472_1] F.* x161[x442_0][x472_0][x472_1])))
    in (let x448 = ((F.neg (x447 F.* (one F./ ((F.sqrt x443) F.* (F.sqrt x443))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x443))))
    in zero)))))))) F.+ (isum1 16 (\ x449_0 -> (isum1 4 (\ x450_0 -> (isum1 4 (\ x451_0 -> (isum2 16 4 (\ x452_0 x452_1 -> (isum1 4 (\ x453_0 -> (isum1 4 (\ x454_0 -> (isum1 16 (\ x455_0 -> (isum1 4 (\ x456_0 -> (isum1 16 (\ x457_0 -> (isum1 16 (\ x458_0 -> (isum1 16 (\ x459_0 -> (isum1 16 (\ x460_0 -> ((if ((x468_0 == x459_0)) then ((F.exp (((isum1 4 (\ x475_0 -> (x143[x459_0][x454_0][x475_0] F.* x146[x468_1][x454_0][x475_0]))) F./ fromi64 2) F.+ mask[x459_0][x468_1])) F.* (F.neg (((if ((x459_0 == x457_0)) then (if ((x460_0 == x458_0)) then ((if ((x457_0 == x455_0)) then (if ((x454_0 == x453_0)) then (if ((x455_0 == x452_0) && (x456_0 == x452_1)) then (if ((x452_0 == x449_0)) then (if ((x452_1 == x451_0)) then (if ((x453_0 == x450_0)) then x240[x449_0][x450_0][x451_0] else zero) else zero) else zero) else zero) else zero) else zero) F.* x149[x458_0][x454_0][x456_0]) else zero) else zero) F.* (F.exp (((isum1 4 (\ x476_0 -> (x143[x459_0][x454_0][x476_0] F.* x146[x460_0][x454_0][x476_0]))) F./ fromi64 2) F.+ mask[x459_0][x460_0]))) F.* (one F./ ((isum1 16 (\ x473_0 -> (F.exp (((isum1 4 (\ x477_0 -> (x143[x459_0][x454_0][x477_0] F.* x146[x473_0][x454_0][x477_0]))) F./ fromi64 2) F.+ mask[x459_0][x473_0])))) F.* (isum1 16 (\ x474_0 -> (F.exp (((isum1 4 (\ x478_0 -> (x143[x459_0][x454_0][x478_0] F.* x146[x474_0][x454_0][x478_0]))) F./ fromi64 2) F.+ mask[x459_0][x474_0]))))))))) else zero) F.+ (if ((x468_0 == x459_0)) then (if ((x468_1 == x460_0)) then ((F.exp (((isum1 4 (\ x480_0 -> (x143[x459_0][x454_0][x480_0] F.* x146[x460_0][x454_0][x480_0]))) F./ fromi64 2) F.+ mask[x459_0][x460_0])) F.* ((if ((x459_0 == x457_0)) then (if ((x460_0 == x458_0)) then ((if ((x457_0 == x455_0)) then (if ((x454_0 == x453_0)) then (if ((x455_0 == x452_0) && (x456_0 == x452_1)) then (if ((x452_0 == x449_0)) then (if ((x452_1 == x451_0)) then (if ((x453_0 == x450_0)) then x240[x449_0][x450_0][x451_0] else zero) else zero) else zero) else zero) else zero) else zero) F.* x149[x458_0][x454_0][x456_0]) else zero) else zero) F.* (one F./ (isum1 16 (\ x479_0 -> (F.exp (((isum1 4 (\ x481_0 -> (x143[x459_0][x454_0][x481_0] F.* x146[x479_0][x454_0][x481_0]))) F./ fromi64 2) F.+ mask[x459_0][x479_0]))))))) else zero) else zero))))))))))))))))))))))))))) F.+ (isum1 16 (\ x461_0 -> (let x462 = ((isum2 4 4 (\ x482_0 x482_1 -> (x78[x461_0][x482_0][x482_1] F.* x78[x461_0][x482_0][x482_1]))) F./ fromi64 16)
    in (let x463 = (one F./ (F.sqrt x462))
    in (let x464 = (imap2 4 4 (\ x483_0 x483_1 -> (x78[x461_0][x483_0][x483_1] F.* x463)))
    in (let x465 = (imap2 4 4 (\ x484_0 x484_1 -> x388[x461_0][x484_0][x484_1]))
    in (let x466 = (isum2 4 4 (\ x485_0 x485_1 -> (x465[x485_0][x485_1] F.* x78[x461_0][x485_0][x485_1])))
    in (let x467 = ((F.neg (x466 F.* (one F./ ((F.sqrt x462) F.* (F.sqrt x462))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x462))))
    in zero)))))))))))
    let dwpe = (imap2 16 16 (\ x500_0 x500_1 -> (((isum1 16 (\ x486_0 -> (let x487 = ((isum2 4 4 (\ x501_0 x501_1 -> (x161[x486_0][x501_0][x501_1] F.* x161[x486_0][x501_0][x501_1]))) F./ fromi64 16)
    in (let x488 = (one F./ (F.sqrt x487))
    in (let x489 = (imap2 4 4 (\ x502_0 x502_1 -> (x161[x486_0][x502_0][x502_1] F.* x488)))
    in (let x490 = (imap2 4 4 (\ x503_0 x503_1 -> x210[x486_0][x503_0][x503_1]))
    in (let x491 = (isum2 4 4 (\ x504_0 x504_1 -> (x490[x504_0][x504_1] F.* x161[x486_0][x504_0][x504_1])))
    in (let x492 = ((F.neg (x491 F.* (one F./ ((F.sqrt x487) F.* (F.sqrt x487))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x487))))
    in zero)))))))) F.+ (isum1 16 (\ x493_0 -> (let x494 = ((isum2 4 4 (\ x505_0 x505_1 -> (x78[x493_0][x505_0][x505_1] F.* x78[x493_0][x505_0][x505_1]))) F./ fromi64 16)
    in (let x495 = (one F./ (F.sqrt x494))
    in (let x496 = (imap2 4 4 (\ x506_0 x506_1 -> (x78[x493_0][x506_0][x506_1] F.* x495)))
    in (let x497 = (imap2 4 4 (\ x507_0 x507_1 -> x388[x493_0][x507_0][x507_1]))
    in (let x498 = (isum2 4 4 (\ x508_0 x508_1 -> (x497[x508_0][x508_1] F.* x78[x493_0][x508_0][x508_1])))
    in (let x499 = ((F.neg (x498 F.* (one F./ ((F.sqrt x494) F.* (F.sqrt x494))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x494))))
    in zero))))))))) F.+ x418[x500_0][x500_1])))
    let dwqry = (imap2 16 16 (\ x523_0 x523_1 -> (((isum1 16 (\ x509_0 -> (let x510 = ((isum2 4 4 (\ x524_0 x524_1 -> (x161[x509_0][x524_0][x524_1] F.* x161[x509_0][x524_0][x524_1]))) F./ fromi64 16)
    in (let x511 = (one F./ (F.sqrt x510))
    in (let x512 = (imap2 4 4 (\ x525_0 x525_1 -> (x161[x509_0][x525_0][x525_1] F.* x511)))
    in (let x513 = (imap2 4 4 (\ x526_0 x526_1 -> x210[x509_0][x526_0][x526_1]))
    in (let x514 = (isum2 4 4 (\ x527_0 x527_1 -> (x513[x527_0][x527_1] F.* x161[x509_0][x527_0][x527_1])))
    in (let x515 = ((F.neg (x514 F.* (one F./ ((F.sqrt x510) F.* (F.sqrt x510))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x510))))
    in zero)))))))) F.+ (isum1 16 (\ x528_0 -> (isum2 4 4 (\ x529_0 x529_1 -> (isum2 4 4 (\ x530_0 x530_1 -> (isum1 4 (\ x531_0 -> (isum1 4 (\ x532_0 -> (isum1 4 (\ x533_0 -> (isum1 4 (\ x534_0 -> (isum2 4 4 (\ x535_0 x535_1 -> (isum1 4 (\ x536_0 -> (isum1 4 (\ x537_0 -> (if (((x523_0 / 4) == x531_0)) then (if (((x523_1 / 4) == x537_0)) then (if ((x537_0 == x536_0)) then (if (((x523_0 % 4) == x535_0) && ((x523_1 % 4) == x535_1)) then (if ((x535_0 == x532_0)) then (if ((x535_1 == x534_0)) then (if ((x536_0 == x533_0)) then (if ((x531_0 == x529_0) && (x532_0 == x529_1)) then (if ((x533_0 == x530_0) && (x534_0 == x530_1)) then (x339[x528_0][x529_0][x529_1] F.* x136[x528_0][x530_0][x530_1]) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero)))))))))))))))))))))) F.+ (isum1 16 (\ x516_0 -> (let x517 = ((isum2 4 4 (\ x538_0 x538_1 -> (x78[x516_0][x538_0][x538_1] F.* x78[x516_0][x538_0][x538_1]))) F./ fromi64 16)
    in (let x518 = (one F./ (F.sqrt x517))
    in (let x519 = (imap2 4 4 (\ x539_0 x539_1 -> (x78[x516_0][x539_0][x539_1] F.* x518)))
    in (let x520 = (imap2 4 4 (\ x540_0 x540_1 -> x388[x516_0][x540_0][x540_1]))
    in (let x521 = (isum2 4 4 (\ x541_0 x541_1 -> (x520[x541_0][x541_1] F.* x78[x516_0][x541_0][x541_1])))
    in (let x522 = ((F.neg (x521 F.* (one F./ ((F.sqrt x517) F.* (F.sqrt x517))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x517))))
    in zero)))))))))))
    let dwkey = (imap2 16 16 (\ x556_0 x556_1 -> (((isum1 16 (\ x542_0 -> (let x543 = ((isum2 4 4 (\ x557_0 x557_1 -> (x161[x542_0][x557_0][x557_1] F.* x161[x542_0][x557_0][x557_1]))) F./ fromi64 16)
    in (let x544 = (one F./ (F.sqrt x543))
    in (let x545 = (imap2 4 4 (\ x558_0 x558_1 -> (x161[x542_0][x558_0][x558_1] F.* x544)))
    in (let x546 = (imap2 4 4 (\ x559_0 x559_1 -> x210[x542_0][x559_0][x559_1]))
    in (let x547 = (isum2 4 4 (\ x560_0 x560_1 -> (x546[x560_0][x560_1] F.* x161[x542_0][x560_0][x560_1])))
    in (let x548 = ((F.neg (x547 F.* (one F./ ((F.sqrt x543) F.* (F.sqrt x543))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x543))))
    in zero)))))))) F.+ (isum1 16 (\ x561_0 -> (isum2 4 4 (\ x562_0 x562_1 -> (isum2 4 4 (\ x563_0 x563_1 -> (isum1 4 (\ x564_0 -> (isum1 4 (\ x565_0 -> (isum1 4 (\ x566_0 -> (isum1 4 (\ x567_0 -> (isum2 4 4 (\ x568_0 x568_1 -> (isum1 4 (\ x569_0 -> (isum1 4 (\ x570_0 -> (if (((x556_0 / 4) == x564_0)) then (if (((x556_1 / 4) == x570_0)) then (if ((x570_0 == x569_0)) then (if (((x556_0 % 4) == x568_0) && ((x556_1 % 4) == x568_1)) then (if ((x568_0 == x565_0)) then (if ((x568_1 == x567_0)) then (if ((x569_0 == x566_0)) then (if ((x564_0 == x562_0) && (x565_0 == x562_1)) then (if ((x566_0 == x563_0) && (x567_0 == x563_1)) then (x284[x561_0][x562_0][x562_1] F.* x136[x561_0][x563_0][x563_1]) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero)))))))))))))))))))))) F.+ (isum1 16 (\ x549_0 -> (let x550 = ((isum2 4 4 (\ x571_0 x571_1 -> (x78[x549_0][x571_0][x571_1] F.* x78[x549_0][x571_0][x571_1]))) F./ fromi64 16)
    in (let x551 = (one F./ (F.sqrt x550))
    in (let x552 = (imap2 4 4 (\ x572_0 x572_1 -> (x78[x549_0][x572_0][x572_1] F.* x551)))
    in (let x553 = (imap2 4 4 (\ x573_0 x573_1 -> x388[x549_0][x573_0][x573_1]))
    in (let x554 = (isum2 4 4 (\ x574_0 x574_1 -> (x553[x574_0][x574_1] F.* x78[x549_0][x574_0][x574_1])))
    in (let x555 = ((F.neg (x554 F.* (one F./ ((F.sqrt x550) F.* (F.sqrt x550))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x550))))
    in zero)))))))))))
    let dwval = (imap2 16 16 (\ x589_0 x589_1 -> (((isum1 16 (\ x575_0 -> (let x576 = ((isum2 4 4 (\ x590_0 x590_1 -> (x161[x575_0][x590_0][x590_1] F.* x161[x575_0][x590_0][x590_1]))) F./ fromi64 16)
    in (let x577 = (one F./ (F.sqrt x576))
    in (let x578 = (imap2 4 4 (\ x591_0 x591_1 -> (x161[x575_0][x591_0][x591_1] F.* x577)))
    in (let x579 = (imap2 4 4 (\ x592_0 x592_1 -> x210[x575_0][x592_0][x592_1]))
    in (let x580 = (isum2 4 4 (\ x593_0 x593_1 -> (x579[x593_0][x593_1] F.* x161[x575_0][x593_0][x593_1])))
    in (let x581 = ((F.neg (x580 F.* (one F./ ((F.sqrt x576) F.* (F.sqrt x576))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x576))))
    in zero)))))))) F.+ (isum1 16 (\ x594_0 -> (isum2 4 4 (\ x595_0 x595_1 -> (isum2 4 4 (\ x596_0 x596_1 -> (isum1 4 (\ x597_0 -> (isum1 4 (\ x598_0 -> (isum1 4 (\ x599_0 -> (isum1 4 (\ x600_0 -> (isum2 4 4 (\ x601_0 x601_1 -> (isum1 4 (\ x602_0 -> (isum1 4 (\ x603_0 -> (if (((x589_0 / 4) == x597_0)) then (if (((x589_1 / 4) == x603_0)) then (if ((x603_0 == x602_0)) then (if (((x589_0 % 4) == x601_0) && ((x589_1 % 4) == x601_1)) then (if ((x601_0 == x598_0)) then (if ((x601_1 == x600_0)) then (if ((x602_0 == x599_0)) then (if ((x597_0 == x595_0) && (x598_0 == x595_1)) then (if ((x599_0 == x596_0) && (x600_0 == x596_1)) then (x254[x594_0][x595_0][x595_1] F.* x136[x594_0][x596_0][x596_1]) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero)))))))))))))))))))))) F.+ (isum1 16 (\ x582_0 -> (let x583 = ((isum2 4 4 (\ x604_0 x604_1 -> (x78[x582_0][x604_0][x604_1] F.* x78[x582_0][x604_0][x604_1]))) F./ fromi64 16)
    in (let x584 = (one F./ (F.sqrt x583))
    in (let x585 = (imap2 4 4 (\ x605_0 x605_1 -> (x78[x582_0][x605_0][x605_1] F.* x584)))
    in (let x586 = (imap2 4 4 (\ x606_0 x606_1 -> x388[x582_0][x606_0][x606_1]))
    in (let x587 = (isum2 4 4 (\ x607_0 x607_1 -> (x586[x607_0][x607_1] F.* x78[x582_0][x607_0][x607_1])))
    in (let x588 = ((F.neg (x587 F.* (one F./ ((F.sqrt x583) F.* (F.sqrt x583))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x583))))
    in zero)))))))))))
    let dwout = (imap2 16 16 (\ x622_0 x622_1 -> (((isum1 16 (\ x608_0 -> (let x609 = ((isum2 4 4 (\ x623_0 x623_1 -> (x161[x608_0][x623_0][x623_1] F.* x161[x608_0][x623_0][x623_1]))) F./ fromi64 16)
    in (let x610 = (one F./ (F.sqrt x609))
    in (let x611 = (imap2 4 4 (\ x624_0 x624_1 -> (x161[x608_0][x624_0][x624_1] F.* x610)))
    in (let x612 = (imap2 4 4 (\ x625_0 x625_1 -> x210[x608_0][x625_0][x625_1]))
    in (let x613 = (isum2 4 4 (\ x626_0 x626_1 -> (x612[x626_0][x626_1] F.* x161[x608_0][x626_0][x626_1])))
    in (let x614 = ((F.neg (x613 F.* (one F./ ((F.sqrt x609) F.* (F.sqrt x609))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x609))))
    in zero)))))))) F.+ (isum1 16 (\ x627_0 -> (isum2 4 4 (\ x628_0 x628_1 -> (isum2 4 4 (\ x629_0 x629_1 -> (isum1 4 (\ x630_0 -> (isum1 4 (\ x631_0 -> (isum1 4 (\ x632_0 -> (isum1 4 (\ x633_0 -> (isum2 4 4 (\ x634_0 x634_1 -> (isum1 4 (\ x635_0 -> (isum1 4 (\ x636_0 -> (if (((x622_0 / 4) == x630_0)) then (if (((x622_1 / 4) == x636_0)) then (if ((x636_0 == x635_0)) then (if (((x622_0 % 4) == x634_0) && ((x622_1 % 4) == x634_1)) then (if ((x634_0 == x631_0)) then (if ((x634_1 == x633_0)) then (if ((x635_0 == x632_0)) then (if ((x630_0 == x628_0) && (x631_0 == x628_1)) then (if ((x632_0 == x629_0) && (x633_0 == x629_1)) then (x227[x627_0][x628_0][x628_1] F.* x152[x627_0][x629_0][x629_1]) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero) else zero)))))))))))))))))))))) F.+ (isum1 16 (\ x615_0 -> (let x616 = ((isum2 4 4 (\ x637_0 x637_1 -> (x78[x615_0][x637_0][x637_1] F.* x78[x615_0][x637_0][x637_1]))) F./ fromi64 16)
    in (let x617 = (one F./ (F.sqrt x616))
    in (let x618 = (imap2 4 4 (\ x638_0 x638_1 -> (x78[x615_0][x638_0][x638_1] F.* x617)))
    in (let x619 = (imap2 4 4 (\ x639_0 x639_1 -> x388[x615_0][x639_0][x639_1]))
    in (let x620 = (isum2 4 4 (\ x640_0 x640_1 -> (x619[x640_0][x640_1] F.* x78[x615_0][x640_0][x640_1])))
    in (let x621 = ((F.neg (x620 F.* (one F./ ((F.sqrt x616) F.* (F.sqrt x616))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x616))))
    in zero)))))))))))
    let dwup = (imap3 64 16 16 (\ x655_0 x655_1 x655_2 -> (((isum1 16 (\ x656_0 -> (isum2 64 16 (\ x657_0 x657_1 -> (isum2 4 4 (\ x658_0 x658_1 -> (if ((x655_0 == x657_0) && (x655_1 == x657_1)) then (if (((x655_2 / 4) == x658_0) && ((x655_2 % 4) == x658_1)) then (x208[x656_0][x657_0][x657_1] F.* x184[x656_0][x658_0][x658_1]) else zero) else zero))))))) F.+ (isum1 16 (\ x641_0 -> (let x642 = ((isum2 4 4 (\ x659_0 x659_1 -> (x161[x641_0][x659_0][x659_1] F.* x161[x641_0][x659_0][x659_1]))) F./ fromi64 16)
    in (let x643 = (one F./ (F.sqrt x642))
    in (let x644 = (imap2 4 4 (\ x660_0 x660_1 -> (x161[x641_0][x660_0][x660_1] F.* x643)))
    in (let x645 = (imap2 4 4 (\ x661_0 x661_1 -> x210[x641_0][x661_0][x661_1]))
    in (let x646 = (isum2 4 4 (\ x662_0 x662_1 -> (x645[x662_0][x662_1] F.* x161[x641_0][x662_0][x662_1])))
    in (let x647 = ((F.neg (x646 F.* (one F./ ((F.sqrt x642) F.* (F.sqrt x642))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x642))))
    in zero))))))))) F.+ (isum1 16 (\ x648_0 -> (let x649 = ((isum2 4 4 (\ x663_0 x663_1 -> (x78[x648_0][x663_0][x663_1] F.* x78[x648_0][x663_0][x663_1]))) F./ fromi64 16)
    in (let x650 = (one F./ (F.sqrt x649))
    in (let x651 = (imap2 4 4 (\ x664_0 x664_1 -> (x78[x648_0][x664_0][x664_1] F.* x650)))
    in (let x652 = (imap2 4 4 (\ x665_0 x665_1 -> x388[x648_0][x665_0][x665_1]))
    in (let x653 = (isum2 4 4 (\ x666_0 x666_1 -> (x652[x666_0][x666_1] F.* x78[x648_0][x666_0][x666_1])))
    in (let x654 = ((F.neg (x653 F.* (one F./ ((F.sqrt x649) F.* (F.sqrt x649))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x649))))
    in zero)))))))))))
    let dwdown = (imap3 16 64 16 (\ x681_0 x681_1 x681_2 -> (((isum1 16 (\ x682_0 -> (isum2 4 4 (\ x683_0 x683_1 -> (isum2 64 16 (\ x684_0 x684_1 -> (isum2 4 4 (\ x685_0 x685_1 -> (isum2 64 16 (\ x686_0 x686_1 -> (isum1 4 (\ x687_0 -> (if (((x681_0 / 4) == x687_0)) then (if ((x681_1 == x686_0) && (x681_2 == x686_1)) then (if ((x687_0 == x685_0) && ((x681_0 % 4) == x685_1)) then (if ((x685_0 == x683_0) && (x685_1 == x683_1)) then (if ((x686_0 == x684_0) && (x686_1 == x684_1)) then (x203[x682_0][x683_0][x683_1] F.* x194[x682_0][x684_0][x684_1]) else zero) else zero) else zero) else zero) else zero))))))))))))) F.+ (isum1 16 (\ x667_0 -> (let x668 = ((isum2 4 4 (\ x688_0 x688_1 -> (x161[x667_0][x688_0][x688_1] F.* x161[x667_0][x688_0][x688_1]))) F./ fromi64 16)
    in (let x669 = (one F./ (F.sqrt x668))
    in (let x670 = (imap2 4 4 (\ x689_0 x689_1 -> (x161[x667_0][x689_0][x689_1] F.* x669)))
    in (let x671 = (imap2 4 4 (\ x690_0 x690_1 -> x210[x667_0][x690_0][x690_1]))
    in (let x672 = (isum2 4 4 (\ x691_0 x691_1 -> (x671[x691_0][x691_1] F.* x161[x667_0][x691_0][x691_1])))
    in (let x673 = ((F.neg (x672 F.* (one F./ ((F.sqrt x668) F.* (F.sqrt x668))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x668))))
    in zero))))))))) F.+ (isum1 16 (\ x674_0 -> (let x675 = ((isum2 4 4 (\ x692_0 x692_1 -> (x78[x674_0][x692_0][x692_1] F.* x78[x674_0][x692_0][x692_1]))) F./ fromi64 16)
    in (let x676 = (one F./ (F.sqrt x675))
    in (let x677 = (imap2 4 4 (\ x693_0 x693_1 -> (x78[x674_0][x693_0][x693_1] F.* x676)))
    in (let x678 = (imap2 4 4 (\ x694_0 x694_1 -> x388[x674_0][x694_0][x694_1]))
    in (let x679 = (isum2 4 4 (\ x695_0 x695_1 -> (x678[x695_0][x695_1] F.* x78[x674_0][x695_0][x695_1])))
    in (let x680 = ((F.neg (x679 F.* (one F./ ((F.sqrt x675) F.* (F.sqrt x675))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x675))))
    in zero)))))))))))
    let dwvoc = (imap2 27 16 (\ x710_0 x710_1 -> (((isum1 16 (\ x711_0 -> (isum1 27 (\ x712_0 -> (isum2 4 4 (\ x713_0 x713_1 -> (if ((x710_0 == x712_0)) then (if (((x710_1 / 4) == x713_0) && ((x710_1 % 4) == x713_1)) then (x131[x711_0][x712_0] F.* x80[x711_0][x713_0][x713_1]) else zero) else zero))))))) F.+ (isum1 16 (\ x696_0 -> (let x697 = ((isum2 4 4 (\ x714_0 x714_1 -> (x161[x696_0][x714_0][x714_1] F.* x161[x696_0][x714_0][x714_1]))) F./ fromi64 16)
    in (let x698 = (one F./ (F.sqrt x697))
    in (let x699 = (imap2 4 4 (\ x715_0 x715_1 -> (x161[x696_0][x715_0][x715_1] F.* x698)))
    in (let x700 = (imap2 4 4 (\ x716_0 x716_1 -> x210[x696_0][x716_0][x716_1]))
    in (let x701 = (isum2 4 4 (\ x717_0 x717_1 -> (x700[x717_0][x717_1] F.* x161[x696_0][x717_0][x717_1])))
    in (let x702 = ((F.neg (x701 F.* (one F./ ((F.sqrt x697) F.* (F.sqrt x697))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x697))))
    in zero))))))))) F.+ (isum1 16 (\ x703_0 -> (let x704 = ((isum2 4 4 (\ x718_0 x718_1 -> (x78[x703_0][x718_0][x718_1] F.* x78[x703_0][x718_0][x718_1]))) F./ fromi64 16)
    in (let x705 = (one F./ (F.sqrt x704))
    in (let x706 = (imap2 4 4 (\ x719_0 x719_1 -> (x78[x703_0][x719_0][x719_1] F.* x705)))
    in (let x707 = (imap2 4 4 (\ x720_0 x720_1 -> x388[x703_0][x720_0][x720_1]))
    in (let x708 = (isum2 4 4 (\ x721_0 x721_1 -> (x707[x721_0][x721_1] F.* x78[x703_0][x721_0][x721_1])))
    in (let x709 = ((F.neg (x708 F.* (one F./ ((F.sqrt x704) F.* (F.sqrt x704))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x704))))
    in zero)))))))))))
    let dwseq = (imap2 16 16 (\ x736_0 x736_1 -> (((isum1 16 (\ x722_0 -> (let x723 = ((isum2 4 4 (\ x737_0 x737_1 -> (x161[x722_0][x737_0][x737_1] F.* x161[x722_0][x737_0][x737_1]))) F./ fromi64 16)
    in (let x724 = (one F./ (F.sqrt x723))
    in (let x725 = (imap2 4 4 (\ x738_0 x738_1 -> (x161[x722_0][x738_0][x738_1] F.* x724)))
    in (let x726 = (imap2 4 4 (\ x739_0 x739_1 -> x210[x722_0][x739_0][x739_1]))
    in (let x727 = (isum2 4 4 (\ x740_0 x740_1 -> (x726[x740_0][x740_1] F.* x161[x722_0][x740_0][x740_1])))
    in (let x728 = ((F.neg (x727 F.* (one F./ ((F.sqrt x723) F.* (F.sqrt x723))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x723))))
    in zero)))))))) F.+ (isum1 16 (\ x729_0 -> (let x730 = ((isum2 4 4 (\ x741_0 x741_1 -> (x78[x729_0][x741_0][x741_1] F.* x78[x729_0][x741_0][x741_1]))) F./ fromi64 16)
    in (let x731 = (one F./ (F.sqrt x730))
    in (let x732 = (imap2 4 4 (\ x742_0 x742_1 -> (x78[x729_0][x742_0][x742_1] F.* x731)))
    in (let x733 = (imap2 4 4 (\ x743_0 x743_1 -> x388[x729_0][x743_0][x743_1]))
    in (let x734 = (isum2 4 4 (\ x744_0 x744_1 -> (x733[x744_0][x744_1] F.* x78[x729_0][x744_0][x744_1])))
    in (let x735 = ((F.neg (x734 F.* (one F./ ((F.sqrt x730) F.* (F.sqrt x730))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x730))))
    in zero))))))))) F.+ x418[x736_0][x736_1])))
    let dtarget = (imap2 16 27 (\ x759_0 x759_1 -> ((((F.neg x64[x759_0]) F.* (F.log ((F.exp x0[x759_0][x759_1]) F.* (one F./ (F.exp x0[x759_0][x759_1]))))) F.+ (isum1 16 (\ x745_0 -> (let x746 = ((isum2 4 4 (\ x761_0 x761_1 -> (x161[x745_0][x761_0][x761_1] F.* x161[x745_0][x761_0][x761_1]))) F./ fromi64 16)
    in (let x747 = (one F./ (F.sqrt x746))
    in (let x748 = (imap2 4 4 (\ x762_0 x762_1 -> (x161[x745_0][x762_0][x762_1] F.* x747)))
    in (let x749 = (imap2 4 4 (\ x763_0 x763_1 -> x210[x745_0][x763_0][x763_1]))
    in (let x750 = (isum2 4 4 (\ x764_0 x764_1 -> (x749[x764_0][x764_1] F.* x161[x745_0][x764_0][x764_1])))
    in (let x751 = ((F.neg (x750 F.* (one F./ ((F.sqrt x746) F.* (F.sqrt x746))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x746))))
    in zero))))))))) F.+ (isum1 16 (\ x752_0 -> (let x753 = ((isum2 4 4 (\ x765_0 x765_1 -> (x78[x752_0][x765_0][x765_1] F.* x78[x752_0][x765_0][x765_1]))) F./ fromi64 16)
    in (let x754 = (one F./ (F.sqrt x753))
    in (let x755 = (imap2 4 4 (\ x766_0 x766_1 -> (x78[x752_0][x766_0][x766_1] F.* x754)))
    in (let x756 = (imap2 4 4 (\ x767_0 x767_1 -> x388[x752_0][x767_0][x767_1]))
    in (let x757 = (isum2 4 4 (\ x768_0 x768_1 -> (x756[x768_0][x768_1] F.* x78[x752_0][x768_0][x768_1])))
    in (let x758 = ((F.neg (x757 F.* (one F./ ((F.sqrt x753) F.* (F.sqrt x753))))) F.* (one F./ ((x63 F.+ x63) F.* (F.sqrt x753))))
    in zero)))))))))))


    in (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dwseq)

  -- def train_gen : (inp: [28][28]real)
  -- -> (k1: [6][5][5]real)
  -- -> (b1: [6]real)
  -- -> (k2: [12][6][5][5]real)
  -- -> (b2: [12]real)
  -- -> (fc: [10][12][4][4]real)
  -- -> (b: [10]real)
  -- -> (target: [10]real)
  -- -> ( [6][5][5]real
  --    , -- ∂k1
  --      [6]real
  --    , -- ∂b1
  --      [12][6][5][5]real
  --    , -- ∂k2
  --      [12]real
  --    , -- ∂b2
  --      [10][12][4][4]real
  --    , -- ∂fc
  --      [10]real
  --    , -- ∂b
  --      real
  --    ) =
  --   -- error
  --   #[unsafe]
  --   \(inp: [28][28]real) (k1: [6][5][5]real) (b1: [6]real) (k2: [12][6][5][5]real) (b2: [12]real) (fc': [10][12][4][4]real) (b: [10]real) (target': [10]real) ->
  --     let fc = map (\a -> unflatten (a :> [12 * 1][4][4]real)) fc' :> [10][12][1][4][4]real
  --     let target = unflatten (unflatten (unflatten (unflatten (target' :> [10 * 1]real) :> [10 * 1][1]real) :> [10 * 1][1][1]real) :> [10 * 1][1][1][1]real) :> [10][1][1][1][1]real

  --     let x0 = (imap3 6 24 24 (\x1_0 x1_1 x1_2 -> ((isum2 5 5 (\x2_0 x2_1 -> (inp[(x2_0 + x1_1)][(x2_1 + x1_2)] F.* k1[x1_0][x2_0][x2_1]))) F.+ b1[x1_0])))
  --     let x3 = (imap3 6 24 24 (\x4_0 x4_1 x4_2 -> (logistics x0[x4_0][x4_1][x4_2])))
  --     let x5 = (imap3 6 12 12 (\x6_0 x6_1 x6_2 -> ((isum2 2 2 (\x7_0 x7_1 -> x3[x6_0][((x6_1 * 2) + x7_0)][((x6_2 * 2) + x7_1)])) F./ fromi64 4)))
  --     let x8 = (imap4 12 1 8 8 (\x9_0 x9_1 x9_2 x9_3 -> ((isum3 6 5 5 (\x10_0 x10_1 x10_2 -> (x5[(x10_0 + x9_1)][(x10_1 + x9_2)][(x10_2 + x9_3)] F.* k2[x9_0][x10_0][x10_1][x10_2]))) F.+ b2[x9_0])))
  --     let x11 = (imap4 12 1 8 8 (\x12_0 x12_1 x12_2 x12_3 -> (logistics x8[x12_0][x12_1][x12_2][x12_3])))
  --     let x13 = (imap4 12 1 4 4 (\x14_0 x14_1 x14_2 x14_3 -> ((isum2 2 2 (\x15_0 x15_1 -> x11[x14_0][x14_1][((x14_2 * 2) + x15_0)][((x14_3 * 2) + x15_1)])) F./ fromi64 4)))
  --     let x16 = (imap5 10 1 1 1 1 (\x17_0 x17_1 x17_2 x17_3 x17_4 -> ((isum4 12 1 4 4 (\x18_0 x18_1 x18_2 x18_3 -> (x13[(x18_0 + x17_1)][(x18_1 + x17_2)][(x18_2 + x17_3)][(x18_3 + x17_4)] F.* fc[x17_0][x18_0][x18_1][x18_2][x18_3]))) F.+ b[x17_0])))
  --     let x19 = (imap5 10 1 1 1 1 (\x20_0 x20_1 x20_2 x20_3 x20_4 -> (logistics x16[x20_0][x20_1][x20_2][x20_3][x20_4])))
  --     let x21 = (isum5 10 1 1 1 1 (\x22_0 x22_1 x22_2 x22_3 x22_4 -> (((target[x22_0][x22_1][x22_2][x22_3][x22_4] F.+ (F.neg x19[x22_0][x22_1][x22_2][x22_3][x22_4])) F.* (target[x22_0][x22_1][x22_2][x22_3][x22_4] F.+ (F.neg x19[x22_0][x22_1][x22_2][x22_3][x22_4]))) F./ fromi64 2)))
  --     let x23 = one
  --     let x24 = (imap5 10 1 1 1 1 (\x25_0 x25_1 x25_2 x25_3 x25_4 -> ((F.neg ((x23 F./ fromi64 2) F.* (target[x25_0][x25_1][x25_2][x25_3][x25_4] F.+ (F.neg x19[x25_0][x25_1][x25_2][x25_3][x25_4])))) F.+ (F.neg ((x23 F./ fromi64 2) F.* (target[x25_0][x25_1][x25_2][x25_3][x25_4] F.+ (F.neg x19[x25_0][x25_1][x25_2][x25_3][x25_4])))))))
  --     let x26 = (imap5 10 1 1 1 1 (\x27_0 x27_1 x27_2 x27_3 x27_4 -> ((x24[x27_0][x27_1][x27_2][x27_3][x27_4] F.* x19[x27_0][x27_1][x27_2][x27_3][x27_4]) F.* (one F.+ (F.neg x19[x27_0][x27_1][x27_2][x27_3][x27_4])))))
  --     let x28 = (imap4 12 1 4 4 (\x31_0 x31_1 x31_2 x31_3 -> (isum1 10 (\x29_0 -> (isum4 12 1 4 4 (\x30_0 x30_1 x30_2 x30_3 -> if (x31_0 >= x30_0 && x31_1 >= x30_1 && x31_2 >= x30_2 && x31_3 >= x30_3 && (x31_0 - x30_0) < 1 && (x31_1 - x30_1) < 1 && (x31_2 - x30_2) < 1 && (x31_3 - x30_3) < 1) then (x26[x29_0][(x31_0 - x30_0)][(x31_1 - x30_1)][(x31_2 - x30_2)][(x31_3 - x30_3)] F.* fc[x29_0][x30_0][x30_1][x30_2][x30_3]) else zero))))))
  --     let x32 = (imap4 12 1 8 8 (\x33_0 x33_1 x33_2 x33_3 -> (x28[x33_0][x33_1][(x33_2 / 2)][(x33_3 / 2)] F./ fromi64 4)))
  --     let x34 = (imap4 12 1 8 8 (\x35_0 x35_1 x35_2 x35_3 -> ((x32[x35_0][x35_1][x35_2][x35_3] F.* x11[x35_0][x35_1][x35_2][x35_3]) F.* (one F.+ (F.neg x11[x35_0][x35_1][x35_2][x35_3])))))
  --     let x36 = (imap3 6 12 12 (\x39_0 x39_1 x39_2 -> (isum1 12 (\x37_0 -> (isum3 6 5 5 (\x38_0 x38_1 x38_2 -> if (x39_0 >= x38_0 && x39_1 >= x38_1 && x39_2 >= x38_2 && (x39_0 - x38_0) < 1 && (x39_1 - x38_1) < 8 && (x39_2 - x38_2) < 8) then (x34[x37_0][(x39_0 - x38_0)][(x39_1 - x38_1)][(x39_2 - x38_2)] F.* k2[x37_0][x38_0][x38_1][x38_2]) else zero))))))
  --     let x40 = (imap3 6 24 24 (\x41_0 x41_1 x41_2 -> (x36[x41_0][(x41_1 / 2)][(x41_2 / 2)] F./ fromi64 4)))
  --     let x42 = (imap3 6 24 24 (\x43_0 x43_1 x43_2 -> ((x40[x43_0][x43_1][x43_2] F.* x3[x43_0][x43_1][x43_2]) F.* (one F.+ (F.neg x3[x43_0][x43_1][x43_2])))))
  --     let dinp = (imap2 28 28 (\x46_0 x46_1 -> (isum1 6 (\x44_0 -> (isum2 5 5 (\x45_0 x45_1 -> if (x46_0 >= x45_0 && x46_1 >= x45_1 && (x46_0 - x45_0) < 24 && (x46_1 - x45_1) < 24) then (x42[x44_0][(x46_0 - x45_0)][(x46_1 - x45_1)] F.* k1[x44_0][x45_0][x45_1]) else zero))))))
  --     let dk1 = (imap3 6 5 5 (\x47_0 x47_1 x47_2 -> (isum2 24 24 (\x48_0 x48_1 -> (x42[x47_0][x48_0][x48_1] F.* inp[(x47_1 + x48_0)][(x47_2 + x48_1)])))))
  --     let db1 = (imap1 6 (\x49_0 -> (isum2 24 24 (\x50_0 x50_1 -> x42[x49_0][x50_0][x50_1]))))
  --     let dk2 = (imap4 12 6 5 5 (\x51_0 x51_1 x51_2 x51_3 -> (isum3 1 8 8 (\x52_0 x52_1 x52_2 -> (x34[x51_0][x52_0][x52_1][x52_2] F.* x5[(x51_1 + x52_0)][(x51_2 + x52_1)][(x51_3 + x52_2)])))))
  --     let db2 = (imap1 12 (\x53_0 -> (isum3 1 8 8 (\x54_0 x54_1 x54_2 -> x34[x53_0][x54_0][x54_1][x54_2]))))
  --     let dfc = (imap5 10 12 1 4 4 (\x55_0 x55_1 x55_2 x55_3 x55_4 -> (isum4 1 1 1 1 (\x56_0 x56_1 x56_2 x56_3 -> (x26[x55_0][x56_0][x56_1][x56_2][x56_3] F.* x13[(x55_1 + x56_0)][(x55_2 + x56_1)][(x55_3 + x56_2)][(x55_4 + x56_3)])))))
  --     let db = (imap1 10 (\x57_0 -> (isum4 1 1 1 1 (\x58_0 x58_1 x58_2 x58_3 -> x26[x57_0][x58_0][x58_1][x58_2][x58_3]))))
  --     --let dtarget = (imap5 10 1 1 1 1 (\ x59_0 x59_1 x59_2 x59_3 x59_4 -> (((x23 F./ fromi64 2) F.* (target[x59_0][x59_1][x59_2][x59_3][x59_4] F.+ (F.neg x19[x59_0][x59_1][x59_2][x59_3][x59_4]))) F.+ ((x23 F./ fromi64 2) F.* (target[x59_0][x59_1][x59_2][x59_3][x59_4] F.+ (F.neg x19[x59_0][x59_1][x59_2][x59_3][x59_4]))))))

  --     let err = x21
  --     let dfc' = map flatten dfc :> [10][12][4][4]real
  --     in (dk1, db1, dk2, db2, dfc', db, err)
}

module nn32 = nn f32

type~ str_pair = ([]u8, []u8) --??

entry convert (imgs_bytes: []u8) (lbls_bytes: []u8) : str_pair =
  (imgs_bytes, lbls_bytes)

type state =
  { k1: [6][5][5]f32
  , b1: [6]f32
  , k2: [12][6][5][5]f32
  , b2: [12]f32
  , fc: [10][12][4][4]f32
  , b: [10]f32
  }

entry iteration [n] (trainings: i64) (batchsize: i64) (rate: f32) (imgs: [n][28][28]f32) (lbls: [n]i8) (s: state) : (state, f32) =
  let gen_target i = imap 10 (\j -> if j == i then 1.0 else 0.0)
  let avg (a: []f32) = nn32.sum a / f32.i64 (length a)
  let (s, err) =
    loop (s, err) = (s, 0.0)
    for i < trainings / batchsize do
      let {k1, b1, k2, b2, fc, b} = s
      -- This is where we call trainings in parallel!
      let r =
        imap batchsize (\j ->
                          let img = imgs[i * batchsize + j]
                          let lbl = gen_target (i64.i8 lbls[i * batchsize + j])
                          in nn32.train_gen img k1 b1 k2 b2 fc b lbl)
      let (bdk1, bdb1, bdk2, bdb2, bdfc, bdb, berr) = unzip7 r
      -- TODO: these should happen in-place, but hopefully this is not
      --       a hotspot, the arrays are rather small.
      let k1' =
        imap3 6 5 5 (\i j k ->
                       k1[i][j][k] - rate * (avg (imap batchsize (\t -> bdk1[t][i][j][k]))))
      let b1' =
        imap1 6 (\i ->
                   b1[i] - rate * (avg (imap batchsize (\t -> bdb1[t][i]))))
      let k2' =
        imap4 12 6 5 5 (\i j k l ->
                          k2[i][j][k][l] - rate * (avg (imap batchsize (\t -> bdk2[t][i][j][k][l]))))
      let b2' =
        imap1 12 (\i ->
                    b2[i] - rate * (avg (imap batchsize (\t -> bdb2[t][i]))))
      let fc' =
        imap4 10 12 4 4 (\i j k l ->
                           fc[i][j][k][l] - rate * (avg (imap batchsize (\t -> bdfc[t][i][j][k][l]))))
      let b' =
        imap1 10 (\i ->
                    b[i] - rate * (avg (imap batchsize (\t -> bdb[t][i]))))
      let err' = err + nn32.sum berr
      in ( {k1 = k1', b1 = b1', k2 = k2', b2 = b2', fc = fc', b = b'}
         , err'
         )
  in (s, err / 10.0 / f32.i64 trainings)

entry initial_state : state =
  let k1 = imap3 6 5 5 (\_ _ _ -> 1.0 / 25.0)
  let b1 = imap1 6 (\_ -> 1.0 / 6.0)
  let k2 = imap4 12 6 5 5 (\_ _ _ _ -> 1.0 / 150.0)
  let b2 = imap1 12 (\_ -> 1.0 / 12.0)
  let fc = imap4 10 12 4 4 (\_ _ _ _ -> 1.0 / 192.0)
  let b = imap1 10 (\_ -> 1.0 / 10.0)
  in {k1, b1, k2, b2, fc, b}
