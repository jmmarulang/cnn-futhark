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

  def imaximum1 : (m: i64) -> (i64 -> real) -> real =
    \m f -> F.maximum (imap1 m f)

  def imaximum2 : (m: i64)
  -> (n: i64)
  -> (i64 -> i64 -> real) -> real =
    \m n f -> F.maximum (imap1 m (\i -> imaximum1 n (f i)))

  def isoftmax1 (m: i64) (f : i64 -> real) : [m]real =
    #[inline]
    let max = imaximum1 m f
    let exps = imap1 m (\x -> F.exp((f x) F.+ F.neg max))
    let scale = isum1 m (\x -> exps[x])
    in imap1 m (\x -> exps[x] F./ scale)

  --==== 2d cases ====--
  def sum2d (a: [][]real) : real =
    sum (map sum a)

  --==== Logistics ====--
  def logistics : real -> real =
    \e -> one F./ (one F.+ F.exp (F.neg e))

  def sgnp : real -> real = \e -> F.sgn (F.max e zero)

  def indicatorp : real -> real = sgnp

  --==== This is the generated function. ====--

  def forward : (mask: [1][16][16]real)
    -> (wpe: [16][16]real)
    -> (wqry: [16][16]real)
    -> (wkey: [16][16]real)
    -> (wval: [16][16]real)
    -> (wout: [16][16]real)
    -> (wup: [64][16]real)
    -> (wdown: [16][64]real)
    -> (wvoc: [27][16]real)
    -> (te: [1][16][16]real)
    -> [1][16][27]real =
    #[unsafe]
    \(mask: [1][16][16]real) (wpe: [16][16]real)
    (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real)
    (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real)
    (wvoc: [27][16]real) (te: [1][16][16]real) -> --(imap2 16 27 (\n m -> one F./ zero))

(let x0 = (imap2 1 16 (\i22 i23 -> (let x24 = (imap1 16 (\i28 -> ((te[i22][i23][i28] F.+ wpe[i23][i28]) F.* (te[i22][i23][i28] F.+ wpe[i23][i28]))))
in (let x25 = ((isum1 16 (\i29 -> x24[i29])) F./ fromi64 16)
in (let x26 = (F.sqrt (x25 F.+ (one F./ fromi64 100000)))
in (imap1 16 (\i27 -> ((te[i22][i23][i27] F.+ wpe[i23][i27]) F.* (one F./ x26)))))))))
in (let x1 = (imap2 1 16 (\i30 i31 -> (imap1 16 (\i32 -> (isum1 16 (\i33 -> (wqry[i32][i33] F.* (let x34 = (imap1 16 (\i37 -> (x0[i30][i31][i37] F.* x0[i30][i31][i37])))
in (let x35 = ((isum1 16 (\i38 -> x34[i38])) F./ fromi64 16)
in (let x36 = (F.sqrt (x35 F.+ (one F./ fromi64 100000)))
in (x0[i30][i31][i33] F.* (one F./ x36))))))))))))
in (let x2 = (imap2 1 16 (\i39 i40 -> (imap1 16 (\i41 -> (isum1 16 (\i42 -> (wkey[i41][i42] F.* (let x43 = (imap1 16 (\i46 -> (x0[i39][i40][i46] F.* x0[i39][i40][i46])))
in (let x44 = ((isum1 16 (\i47 -> x43[i47])) F./ fromi64 16)
in (let x45 = (F.sqrt (x44 F.+ (one F./ fromi64 100000)))
in (x0[i39][i40][i42] F.* (one F./ x45))))))))))))
in (let x3 = (imap2 1 16 (\i48 i49 -> (imap1 16 (\i50 -> (isum1 16 (\i51 -> (wval[i50][i51] F.* (let x52 = (imap1 16 (\i55 -> (x0[i48][i49][i55] F.* x0[i48][i49][i55])))
in (let x53 = ((isum1 16 (\i56 -> x52[i56])) F./ fromi64 16)
in (let x54 = (F.sqrt (x53 F.+ (one F./ fromi64 100000)))
in (x0[i48][i49][i51] F.* (one F./ x54))))))))))))
in (let x4 = (imap2 1 16 (\i57 i58 -> (imap1 4 (\i59 -> (imap1 4 (\i60 -> x1[i57][i58][((i59 * 4) + i60)]))))))
in (let x5 = (imap2 1 16 (\i61 i62 -> (imap1 4 (\i63 -> (imap1 4 (\i64 -> x2[i61][i62][((i63 * 4) + i64)]))))))
in (let x6 = (imap2 1 16 (\i65 i66 -> (imap1 4 (\i67 -> (imap1 4 (\i68 -> x3[i65][i66][((i67 * 4) + i68)]))))))
in (let x7 = (imap1 1 (\i69 -> (imap1 4 (\i70 -> (imap1 16 (\i71 -> (imap1 4 (\i72 -> x4[i69][i71][i70][i72]))))))))
in (let x8 = (imap1 1 (\i73 -> (imap1 4 (\i74 -> (imap1 16 (\i75 -> (imap1 4 (\i76 -> x5[i73][i75][i74][i76]))))))))
in (let x9 = (imap1 1 (\i77 -> (imap1 4 (\i78 -> (imap1 16 (\i79 -> (imap1 4 (\i80 -> x6[i77][i79][i78][i80]))))))))
in (let x10 = (imap2 1 4 (\i81 i82 -> (imap1 16 (\i83 -> (imap1 16 (\i84 -> (isum1 4 (\i85 -> (x7[i81][i82][i83][i85] F.* x8[i81][i82][i84][i85])))))))))
in (let x11 = (imap1 1 (\i86 -> (imap1 4 (\i87 -> (imap2 16 16 (\i88 i89 -> ((x10[i86][i87][i88][i89] F./ fromi64 2) F.+ mask[i86][i88][i89])))))))
in (let x12 = (imap2 1 4 (\i90 i91 -> (imap1 16 (\i92 -> (let x94 = (isoftmax1 16 (\i93 -> x11[i90][i91][i92][i93]))
in (imap1 16 (\i95 -> x94[i95])))))))
in (let x13 = (imap2 1 4 (\i96 i97 -> (imap1 16 (\i98 -> (imap1 4 (\i99 -> (isum1 16 (\i100 -> (x12[i96][i97][i98][i100] F.* x9[i96][i97][i100][i99])))))))))
in (let x14 = (imap1 1 (\i101 -> (imap1 16 (\i102 -> (imap1 4 (\i103 -> (imap1 4 (\i104 -> x13[i101][i103][i102][i104]))))))))
in (let x15 = (imap2 1 16 (\i105 i106 -> (imap1 16 (\i107 -> x14[i105][i106][(i107 / 4)][(i107 % 4)]))))
in (let x16 = (imap2 1 16 (\i108 i109 -> (imap1 16 (\i110 -> (isum1 16 (\i111 -> (wout[i110][i111] F.* x15[i108][i109][i111])))))))
in (let x17 = (imap3 1 16 16 (\i112 i113 i114 -> (x0[i112][i113][i114] F.+ x16[i112][i113][i114])))
in (let x18 = (imap2 1 16 (\i115 i116 -> (imap1 16 (\i117 -> (isum1 64 (\i118 -> (wdown[i117][i118] F.* (F.max (isum1 16 (\i119 -> (wup[i118][i119] F.* (let x120 = (imap1 16 (\i123 -> (x17[i115][i116][i123] F.* x17[i115][i116][i123])))
in (let x121 = ((isum1 16 (\i124 -> x120[i124])) F./ fromi64 16)
in (let x122 = (F.sqrt (x121 F.+ (one F./ fromi64 100000)))
in (x17[i115][i116][i119] F.* (one F./ x122)))))))) zero))))))))
in (let x19 = (imap3 1 16 16 (\i125 i126 i127 -> (x17[i125][i126][i127] F.+ x18[i125][i126][i127])))
in (imap2 1 16 (\i20 i21 -> (imap1 27 (\i128 -> (isum1 16 (\i129 -> (wvoc[i128][i129] F.* x19[i20][i21][i129])))))))))))))))))))))))))))


def loss : (mask: [1][16][16]real)
    -> (wpe: [16][16]real)
    -> (wqry: [16][16]real)
    -> (wkey: [16][16]real)
    -> (wval: [16][16]real)
    -> (wout: [16][16]real)
    -> (wup: [64][16]real)
    -> (wdown: [16][64]real)
    -> (wvoc: [27][16]real)
    -> (te: [1][16][16]real)
    -> (target: [1][16][27]real)
    -> real =
    #[unsafe]
    \(mask: [1][16][16]real) (wpe: [16][16]real)
    (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real)
    (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real)
    (wvoc: [27][16]real) (te: [1][16][16]real) (target: [1][16][27]real) ->

(let x0 = (imap2 1 16 (\i24 i25 -> (let x26 = (imap1 16 (\i30 -> ((te[i24][i25][i30] F.+ wpe[i25][i30]) F.* (te[i24][i25][i30] F.+ wpe[i25][i30]))))
in (let x27 = ((isum1 16 (\i31 -> x26[i31])) F./ fromi64 16)
in (let x28 = (F.sqrt (x27 F.+ (one F./ fromi64 100000)))
in (imap1 16 (\i29 -> ((te[i24][i25][i29] F.+ wpe[i25][i29]) F.* (one F./ x28)))))))))
in (let x1 = (imap2 1 16 (\i32 i33 -> (imap1 16 (\i34 -> (isum1 16 (\i35 -> (wqry[i34][i35] F.* (let x36 = (imap1 16 (\i39 -> (x0[i32][i33][i39] F.* x0[i32][i33][i39])))
in (let x37 = ((isum1 16 (\i40 -> x36[i40])) F./ fromi64 16)
in (let x38 = (F.sqrt (x37 F.+ (one F./ fromi64 100000)))
in (x0[i32][i33][i35] F.* (one F./ x38))))))))))))
in (let x2 = (imap2 1 16 (\i41 i42 -> (imap1 16 (\i43 -> (isum1 16 (\i44 -> (wkey[i43][i44] F.* (let x45 = (imap1 16 (\i48 -> (x0[i41][i42][i48] F.* x0[i41][i42][i48])))
in (let x46 = ((isum1 16 (\i49 -> x45[i49])) F./ fromi64 16)
in (let x47 = (F.sqrt (x46 F.+ (one F./ fromi64 100000)))
in (x0[i41][i42][i44] F.* (one F./ x47))))))))))))
in (let x3 = (imap2 1 16 (\i50 i51 -> (imap1 16 (\i52 -> (isum1 16 (\i53 -> (wval[i52][i53] F.* (let x54 = (imap1 16 (\i57 -> (x0[i50][i51][i57] F.* x0[i50][i51][i57])))
in (let x55 = ((isum1 16 (\i58 -> x54[i58])) F./ fromi64 16)
in (let x56 = (F.sqrt (x55 F.+ (one F./ fromi64 100000)))
in (x0[i50][i51][i53] F.* (one F./ x56))))))))))))
in (let x4 = (imap2 1 16 (\i59 i60 -> (imap1 4 (\i61 -> (imap1 4 (\i62 -> x1[i59][i60][((i61 * 4) + i62)]))))))
in (let x5 = (imap2 1 16 (\i63 i64 -> (imap1 4 (\i65 -> (imap1 4 (\i66 -> x2[i63][i64][((i65 * 4) + i66)]))))))
in (let x6 = (imap2 1 16 (\i67 i68 -> (imap1 4 (\i69 -> (imap1 4 (\i70 -> x3[i67][i68][((i69 * 4) + i70)]))))))
in (let x7 = (imap1 1 (\i71 -> (imap1 4 (\i72 -> (imap1 16 (\i73 -> (imap1 4 (\i74 -> x4[i71][i73][i72][i74]))))))))
in (let x8 = (imap1 1 (\i75 -> (imap1 4 (\i76 -> (imap1 16 (\i77 -> (imap1 4 (\i78 -> x5[i75][i77][i76][i78]))))))))
in (let x9 = (imap1 1 (\i79 -> (imap1 4 (\i80 -> (imap1 16 (\i81 -> (imap1 4 (\i82 -> x6[i79][i81][i80][i82]))))))))
in (let x10 = (imap2 1 4 (\i83 i84 -> (imap1 16 (\i85 -> (imap1 16 (\i86 -> (isum1 4 (\i87 -> (x7[i83][i84][i85][i87] F.* x8[i83][i84][i86][i87])))))))))
in (let x11 = (imap1 1 (\i88 -> (imap1 4 (\i89 -> (imap2 16 16 (\i90 i91 -> ((x10[i88][i89][i90][i91] F./ fromi64 2) F.+ mask[i88][i90][i91])))))))
in (let x12 = (imap2 1 4 (\i92 i93 -> (imap1 16 (\i94 -> (let x96 = (isoftmax1 16 (\i95 -> x11[i92][i93][i94][i95]))
in (imap1 16 (\i97 -> x96[i97])))))))
in (let x13 = (imap2 1 4 (\i98 i99 -> (imap1 16 (\i100 -> (imap1 4 (\i101 -> (isum1 16 (\i102 -> (x12[i98][i99][i100][i102] F.* x9[i98][i99][i102][i101])))))))))
in (let x14 = (imap1 1 (\i103 -> (imap1 16 (\i104 -> (imap1 4 (\i105 -> (imap1 4 (\i106 -> x13[i103][i105][i104][i106]))))))))
in (let x15 = (imap2 1 16 (\i107 i108 -> (imap1 16 (\i109 -> x14[i107][i108][(i109 / 4)][(i109 % 4)]))))
in (let x16 = (imap2 1 16 (\i110 i111 -> (imap1 16 (\i112 -> (isum1 16 (\i113 -> (wout[i112][i113] F.* x15[i110][i111][i113])))))))
in (let x17 = (imap3 1 16 16 (\i114 i115 i116 -> (x0[i114][i115][i116] F.+ x16[i114][i115][i116])))
in (let x18 = (imap2 1 16 (\i117 i118 -> (imap1 16 (\i119 -> (isum1 64 (\i120 -> (wdown[i119][i120] F.* (F.max (isum1 16 (\i121 -> (wup[i120][i121] F.* (let x122 = (imap1 16 (\i125 -> (x17[i117][i118][i125] F.* x17[i117][i118][i125])))
in (let x123 = ((isum1 16 (\i126 -> x122[i126])) F./ fromi64 16)
in (let x124 = (F.sqrt (x123 F.+ (one F./ fromi64 100000)))
in (x17[i117][i118][i121] F.* (one F./ x124)))))))) zero))))))))
in (let x19 = (imap3 1 16 16 (\i127 i128 i129 -> (x17[i127][i128][i129] F.+ x18[i127][i128][i129])))
in (let x20 = (imap2 1 16 (\i130 i131 -> (imap1 27 (\i132 -> (isum1 16 (\i133 -> (wvoc[i132][i133] F.* x19[i130][i131][i133])))))))
in (let x21 = (imap2 1 16 (\i134 i135 -> (let x136 = (imap1 27 (\i140 -> (F.log (let x139 = (isoftmax1 27 (\i138 -> x20[i134][i135][i138]))
in x139[i140]))))
in (F.neg (isum1 27 (\i137 -> (x136[i137] F.* target[i134][i135][i137])))))))
in ((isum2 1 16 (\i22 i23 -> x21[i22][i23])) F./ fromi64 16)))))))))))))))))))))))


  def grad_loss : (mask: [1][16][16]real)
    -> (wpe: [16][16]real)
    -> (wqry: [16][16]real)
    -> (wkey: [16][16]real)
    -> (wval: [16][16]real)
    -> (wout: [16][16]real)
    -> (wup: [64][16]real)
    -> (wdown: [16][64]real)
    -> (wvoc: [27][16]real)
    -> (te: [1][16][16]real)
    -> (target: [1][16][27]real)
    -> ([16][16]real, -- dwpe
        [16][16]real, -- dwqry
        [16][16]real, -- dwkey
        [16][16]real, -- dwval
        [16][16]real, -- dwout
        [64][16]real, -- dwup
        [16][64]real, -- dwdown
        [27][16]real, -- dwvoc
        [1][16][16]real -- dte
        ) =
    #[unsafe]
    \(mask: [1][16][16]real) (wpe: [16][16]real)
    (wqry: [16][16]real) (wkey: [16][16]real) (wval: [16][16]real)
    (wout: [16][16]real) (wup: [64][16]real) (wdown: [16][64]real)
    (wvoc: [27][16]real) (te: [1][16][16]real) (target: [1][16][27]real) ->
--     -- (wpe, wqry, wkey, wval, wout, wup, wdown, wvoc, wseq)

let x0 = (imap1 1 (\i1 -> (imap2 16 16 (\i2 i3 -> (te[i1][i2][i3] F.+ wpe[i2][i3])))))
let x4 = (imap3 1 16 16 (\i5 i6 i7 -> (x0[i5][i6][i7] F.* x0[i5][i6][i7])))
let x8 = (imap2 1 16 (\i9 i10 -> ((isum1 16 (\i11 -> x4[i9][i10][i11])) F./ fromi64 16)))
let x12 = (imap2 1 16 (\i13 i14 -> (F.sqrt (x8[i13][i14] F.+ (one F./ fromi64 100000)))))
let x15 = (imap2 1 16 (\i16 i17 -> (imap1 16 (\i18 -> (x0[i16][i17][i18] F.* (one F./ x12[i16][i17]))))))
let x19 = (imap3 1 16 16 (\i20 i21 i22 -> (x15[i20][i21][i22] F.* x15[i20][i21][i22])))
let x23 = (imap2 1 16 (\i24 i25 -> ((isum1 16 (\i26 -> x19[i24][i25][i26])) F./ fromi64 16)))
let x27 = (imap2 1 16 (\i28 i29 -> (F.sqrt (x23[i28][i29] F.+ (one F./ fromi64 100000)))))
let x30 = (imap2 1 16 (\i31 i32 -> (imap1 16 (\i33 -> (x15[i31][i32][i33] F.* (one F./ x27[i31][i32]))))))
let x34 = (imap2 1 16 (\i35 i36 -> (imap1 16 (\i37 -> (isum1 16 (\i38 -> (wqry[i37][i38] F.* x30[i35][i36][i38])))))))
let x39 = (imap2 1 16 (\i40 i41 -> (imap1 16 (\i42 -> (isum1 16 (\i43 -> (wkey[i42][i43] F.* x30[i40][i41][i43])))))))
let x44 = (imap2 1 16 (\i45 i46 -> (imap1 16 (\i47 -> (isum1 16 (\i48 -> (wval[i47][i48] F.* x30[i45][i46][i48])))))))
let x49 = (imap2 1 16 (\i50 i51 -> (imap1 4 (\i52 -> (imap1 4 (\i53 -> x34[i50][i51][((i52 * 4) + i53)]))))))
let x54 = (imap2 1 16 (\i55 i56 -> (imap1 4 (\i57 -> (imap1 4 (\i58 -> x39[i55][i56][((i57 * 4) + i58)]))))))
let x59 = (imap2 1 16 (\i60 i61 -> (imap1 4 (\i62 -> (imap1 4 (\i63 -> x44[i60][i61][((i62 * 4) + i63)]))))))
let x64 = (imap1 1 (\i65 -> (imap1 4 (\i66 -> (imap1 16 (\i67 -> (imap1 4 (\i68 -> x49[i65][i67][i66][i68]))))))))
let x69 = (imap1 1 (\i70 -> (imap1 4 (\i71 -> (imap1 16 (\i72 -> (imap1 4 (\i73 -> x54[i70][i72][i71][i73]))))))))
let x74 = (imap1 1 (\i75 -> (imap1 4 (\i76 -> (imap1 16 (\i77 -> (imap1 4 (\i78 -> x59[i75][i77][i76][i78]))))))))
let x79 = (imap2 1 4 (\i80 i81 -> (imap1 16 (\i82 -> (imap1 16 (\i83 -> (isum1 4 (\i84 -> (x64[i80][i81][i82][i84] F.* x69[i80][i81][i83][i84])))))))))
let x85 = (imap1 1 (\i86 -> (imap1 4 (\i87 -> (imap2 16 16 (\i88 i89 -> ((x79[i86][i87][i88][i89] F./ fromi64 2) F.+ mask[i86][i88][i89])))))))
let x90 = (imap2 1 4 (\i91 i92 -> (imap1 16 (\i93 -> (let x95 = (isoftmax1 16 (\i94 -> x85[i91][i92][i93][i94]))
in (imap1 16 (\i96 -> x95[i96])))))))
let x97 = (imap2 1 4 (\i98 i99 -> (imap1 16 (\i100 -> (imap1 4 (\i101 -> (isum1 16 (\i102 -> (x90[i98][i99][i100][i102] F.* x74[i98][i99][i102][i101])))))))))
let x103 = (imap1 1 (\i104 -> (imap1 16 (\i105 -> (imap1 4 (\i106 -> (imap1 4 (\i107 -> x97[i104][i106][i105][i107]))))))))
let x108 = (imap2 1 16 (\i109 i110 -> (imap1 16 (\i111 -> x103[i109][i110][(i111 / 4)][(i111 % 4)]))))
let x112 = (imap2 1 16 (\i113 i114 -> (imap1 16 (\i115 -> (isum1 16 (\i116 -> (wout[i115][i116] F.* x108[i113][i114][i116])))))))
let x117 = (imap3 1 16 16 (\i118 i119 i120 -> (x30[i118][i119][i120] F.+ x112[i118][i119][i120])))
let x121 = (imap3 1 16 16 (\i122 i123 i124 -> (x117[i122][i123][i124] F.* x117[i122][i123][i124])))
let x125 = (imap2 1 16 (\i126 i127 -> ((isum1 16 (\i128 -> x121[i126][i127][i128])) F./ fromi64 16)))
let x129 = (imap2 1 16 (\i130 i131 -> (F.sqrt (x125[i130][i131] F.+ (one F./ fromi64 100000)))))
let x132 = (imap2 1 16 (\i133 i134 -> (imap1 16 (\i135 -> (x117[i133][i134][i135] F.* (one F./ x129[i133][i134]))))))
let x136 = (imap2 1 16 (\i137 i138 -> (imap1 64 (\i139 -> (isum1 16 (\i140 -> (wup[i139][i140] F.* x132[i137][i138][i140])))))))
let x141 = (imap3 1 16 64 (\i142 i143 i144 -> (F.max x136[i142][i143][i144] zero)))
let x145 = (imap2 1 16 (\i146 i147 -> (imap1 16 (\i148 -> (isum1 64 (\i149 -> (wdown[i148][i149] F.* x141[i146][i147][i149])))))))
let x150 = (imap3 1 16 16 (\i151 i152 i153 -> (x117[i151][i152][i153] F.+ x145[i151][i152][i153])))
let x154 = (imap2 1 16 (\i155 i156 -> (imap1 27 (\i157 -> (isum1 16 (\i158 -> (wvoc[i157][i158] F.* x150[i155][i156][i158])))))))
let x159 = (imap2 1 16 (\i160 i161 -> (one F./ fromi64 16)))
let x162 = (imap2 1 16 (\i163 i164 -> (imap1 27 (\i167 -> (F.log (let x166 = (isoftmax1 27 (\i165 -> x154[i163][i164][i165]))
in x166[i167]))))))
let x168 = (imap2 1 16 (\i169 i170 -> (imap1 27 (\i171 -> ((F.neg x159[i169][i170]) F.* target[i169][i170][i171])))))
let x172 = (imap2 1 16 (\i173 i174 -> (let x176 = (isoftmax1 27 (\i175 -> x154[i173][i174][i175]))
in (imap1 27 (\i177 -> x176[i177])))))
let x178 = (imap2 1 16 (\i179 i180 -> (imap1 27 (\i183 -> (x168[i179][i180][i183] F.* (one F./ (let x182 = (isoftmax1 27 (\i181 -> x154[i179][i180][i181]))
in x182[i183])))))))
let x184 = (imap2 1 16 (\i185 i186 -> (isum1 27 (\i187 -> (x178[i185][i186][i187] F.* x172[i185][i186][i187])))))
let x188 = (imap2 1 16 (\i189 i190 -> (imap1 27 (\i191 -> (x172[i189][i190][i191] F.* (x178[i189][i190][i191] F.+ (F.neg x184[i189][i190])))))))
let x192 = (imap2 1 16 (\i193 i194 -> (imap1 16 (\i195 -> (isum1 27 (\i196 -> (wvoc[i196][i195] F.* x188[i193][i194][i196])))))))
let x197 = (imap2 1 16 (\i198 i199 -> (imap1 64 (\i200 -> (isum1 16 (\i201 -> (wdown[i201][i200] F.* x192[i198][i199][i201])))))))
let x202 = (imap3 1 16 64 (\i203 i204 i205 -> ((indicatorp x136[i203][i204][i205]) F.* x197[i203][i204][i205])))
let x206 = (imap2 1 16 (\i207 i208 -> (imap1 16 (\i209 -> (isum1 64 (\i210 -> (wup[i210][i209] F.* x202[i207][i208][i210])))))))
let x211 = (imap2 1 16 (\i212 i213 -> (isum1 16 (\i214 -> (F.neg ((one F./ x129[i212][i213]) F.* ((x117[i212][i213][i214] F.* x206[i212][i213][i214]) F.* (one F./ x129[i212][i213]))))))))
let x215 = (imap2 1 16 (\i216 i217 -> (x211[i216][i217] F.* (one F./ ((one F.+ one) F.* x129[i216][i217])))))
let x218 = (imap2 1 16 (\i219 i220 -> (imap1 16 (\i221 -> (x215[i219][i220] F./ fromi64 16)))))
let x222 = (imap2 1 16 (\i223 i224 -> (imap1 16 (\i225 -> (((x192[i223][i224][i225] F.+ (x206[i223][i224][i225] F.* (one F./ x129[i223][i224]))) F.+ (x117[i223][i224][i225] F.* x218[i223][i224][i225])) F.+ (x218[i223][i224][i225] F.* x117[i223][i224][i225]))))))
let x226 = (imap2 1 16 (\i227 i228 -> (imap1 16 (\i229 -> (isum1 16 (\i230 -> (wout[i230][i229] F.* x222[i227][i228][i230])))))))
let x231 = (imap2 1 16 (\i232 i233 -> (imap1 4 (\i234 -> (imap1 4 (\i235 -> x226[i232][i233][((i234 * 4) + i235)]))))))
let x236 = (imap1 1 (\i237 -> (imap1 4 (\i238 -> (imap1 16 (\i239 -> (imap1 4 (\i240 -> x231[i237][i239][i238][i240]))))))))
let x241 = (imap2 1 4 (\i242 i243 -> (imap1 16 (\i244 -> (imap1 16 (\i245 -> (isum1 4 (\i246 -> (x236[i242][i243][i244][i246] F.* x74[i242][i243][i245][i246])))))))))
let x247 = (imap2 1 4 (\i248 i249 -> (imap1 16 (\i250 -> (imap1 16 (\i251 -> x241[i248][i249][i250][i251]))))))
let x252 = (imap2 1 4 (\i253 i254 -> (imap1 16 (\i255 -> (isum1 16 (\i256 -> (x247[i253][i254][i255][i256] F.* x90[i253][i254][i255][i256])))))))
let x257 = (imap2 1 4 (\i258 i259 -> (imap1 16 (\i260 -> (imap1 16 (\i261 -> (x90[i258][i259][i260][i261] F.* (x247[i258][i259][i260][i261] F.+ (F.neg x252[i258][i259][i260])))))))))
let x262 = (imap4 1 4 16 16 (\i263 i264 i265 i266 -> (x257[i263][i264][i265][i266] F./ fromi64 2)))
let x267 = (imap2 1 4 (\i268 i269 -> (imap1 16 (\i270 -> (imap1 4 (\i271 -> (isum1 16 (\i272 -> (x90[i268][i269][i272][i270] F.* x236[i268][i269][i272][i271])))))))))
let x273 = (imap2 1 4 (\i274 i275 -> (imap1 16 (\i276 -> (imap1 4 (\i277 -> (isum1 16 (\i278 -> (x64[i274][i275][i278][i277] F.* x262[i274][i275][i278][i276])))))))))
let x279 = (imap2 1 4 (\i280 i281 -> (imap1 16 (\i282 -> (imap1 4 (\i283 -> (isum1 16 (\i284 -> (x262[i280][i281][i282][i284] F.* x69[i280][i281][i284][i283])))))))))
let x285 = (imap1 1 (\i286 -> (imap1 16 (\i287 -> (imap1 4 (\i288 -> (imap1 4 (\i289 -> x267[i286][i288][i287][i289]))))))))
let x290 = (imap1 1 (\i291 -> (imap1 16 (\i292 -> (imap1 4 (\i293 -> (imap1 4 (\i294 -> x273[i291][i293][i292][i294]))))))))
let x295 = (imap1 1 (\i296 -> (imap1 16 (\i297 -> (imap1 4 (\i298 -> (imap1 4 (\i299 -> x279[i296][i298][i297][i299]))))))))
let x300 = (imap2 1 16 (\i301 i302 -> (imap1 16 (\i303 -> x285[i301][i302][(i303 / 4)][(i303 % 4)]))))
let x304 = (imap2 1 16 (\i305 i306 -> (imap1 16 (\i307 -> x290[i305][i306][(i307 / 4)][(i307 % 4)]))))
let x308 = (imap2 1 16 (\i309 i310 -> (imap1 16 (\i311 -> x295[i309][i310][(i311 / 4)][(i311 % 4)]))))
let x312 = (imap2 1 16 (\i313 i314 -> (imap1 16 (\i315 -> (((x222[i313][i314][i315] F.+ (isum1 16 (\i316 -> (wval[i316][i315] F.* x300[i313][i314][i316])))) F.+ (isum1 16 (\i317 -> (wkey[i317][i315] F.* x304[i313][i314][i317])))) F.+ (isum1 16 (\i318 -> (wqry[i318][i315] F.* x308[i313][i314][i318]))))))))
let x319 = (imap2 1 16 (\i320 i321 -> (isum1 16 (\i322 -> (F.neg ((one F./ x27[i320][i321]) F.* ((x15[i320][i321][i322] F.* x312[i320][i321][i322]) F.* (one F./ x27[i320][i321]))))))))
let x323 = (imap2 1 16 (\i324 i325 -> (x319[i324][i325] F.* (one F./ ((one F.+ one) F.* x27[i324][i325])))))
let x326 = (imap2 1 16 (\i327 i328 -> (imap1 16 (\i329 -> (x323[i327][i328] F./ fromi64 16)))))
let x330 = (imap2 1 16 (\i331 i332 -> (imap1 16 (\i333 -> (((x312[i331][i332][i333] F.* (one F./ x27[i331][i332])) F.+ (x15[i331][i332][i333] F.* x326[i331][i332][i333])) F.+ (x326[i331][i332][i333] F.* x15[i331][i332][i333]))))))
let x334 = (imap2 1 16 (\i335 i336 -> (isum1 16 (\i337 -> (F.neg ((one F./ x12[i335][i336]) F.* ((x0[i335][i336][i337] F.* x330[i335][i336][i337]) F.* (one F./ x12[i335][i336]))))))))
let x338 = (imap2 1 16 (\i339 i340 -> (x334[i339][i340] F.* (one F./ ((one F.+ one) F.* x12[i339][i340])))))
let x341 = (imap2 1 16 (\i342 i343 -> (imap1 16 (\i344 -> (x338[i342][i343] F./ fromi64 16)))))
let x345 = (imap2 1 16 (\i346 i347 -> (imap1 16 (\i348 -> (((x330[i346][i347][i348] F.* (one F./ x12[i346][i347])) F.+ (x0[i346][i347][i348] F.* x341[i346][i347][i348])) F.+ (x341[i346][i347][i348] F.* x0[i346][i347][i348]))))))

let dwpe = (imap2 16 16 (\i350 i351 -> (isum1 1 (\i349 -> x345[i349][i350][i351]))))
let dwqry = (imap1 16 (\i352 -> (imap1 16 (\i353 -> (isum2 1 16 (\i354 i355 -> (x308[i354][i355][i352] F.* x30[i354][i355][i353])))))))
let dwkey = (imap1 16 (\i356 -> (imap1 16 (\i357 -> (isum2 1 16 (\i358 i359 -> (x304[i358][i359][i356] F.* x30[i358][i359][i357])))))))
let dwval = (imap1 16 (\i360 -> (imap1 16 (\i361 -> (isum2 1 16 (\i362 i363 -> (x300[i362][i363][i360] F.* x30[i362][i363][i361])))))))
let dwout = (imap1 16 (\i364 -> (imap1 16 (\i365 -> (isum2 1 16 (\i366 i367 -> (x222[i366][i367][i364] F.* x108[i366][i367][i365])))))))
let dwup = (imap1 64 (\i368 -> (imap1 16 (\i369 -> (isum2 1 16 (\i370 i371 -> (x202[i370][i371][i368] F.* x132[i370][i371][i369])))))))
let dwdown = (imap1 16 (\i372 -> (imap1 64 (\i373 -> (isum2 1 16 (\i374 i375 -> (x192[i374][i375][i372] F.* x141[i374][i375][i373])))))))
let dwvoc = (imap1 27 (\i376 -> (imap1 16 (\i377 -> (isum2 1 16 (\i378 i379 -> (x188[i378][i379][i376] F.* x150[i378][i379][i377])))))))
let dmask = (imap1 1 (\i380 -> (imap2 16 16 (\i382 i383 -> (isum1 4 (\i381 -> x257[i380][i381][i382][i383]))))))
let dte = (imap3 1 16 16 (\i384 i385 i386 -> x345[i384][i385][i386]))
let dtarget = (imap2 1 16 (\i387 i388 -> (imap1 27 (\i389 -> (F.neg (x162[i387][i388][i389] F.* x159[i387][i388]))))))

in (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dte)
}


module nn64 = nn f32

type params = {
  wte:   [27][16]f32, -- token embeddings
  wpe:   [16][16]f32, -- position embeddings
  wqry:  [16][16]f32, -- query weights
  wkey:  [16][16]f32, -- key weights
  wval:  [16][16]f32, -- value weights
  wout:  [16][16]f32, -- output weights
  wup:   [64][16]f32, -- MLP up-projection
  wdown: [16][64]f32, -- MLP down-projection
  wvoc:  [27][16]f32  -- output projection
}

entry to_params (wte: [27][16]f32)  (wpe: [16][16]f32)
    (wqry: [16][16]f32) (wkey: [16][16]f32) (wval: [16][16]f32)
    (wout: [16][16]f32) (wup: [64][16]f32) (wdown: [16][64]f32)
    (wvoc: [27][16]f32) : params =
    {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc}

def from_params (p : params) :
  (
  [27][16]f32, -- dwte
  [16][16]f32, -- dwpe
  [16][16]f32, -- dwqry
  [16][16]f32, -- dwkey
  [16][16]f32, -- dwval
  [16][16]f32, -- dwout
  [64][16]f32, -- dwup
  [16][64]f32, -- dwdown
  [27][16]f32, -- dwvoc
  ) =
  let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
  in (wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc)

entry forward (p : params) (seqs : [1][16]i64) (masks : [1][16][16]f32) : [1][16][27]f32 =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let te = (imap3 1 16 16 (\b m n -> wte[seqs[b][m]][n]))
   in nn64.forward masks wpe wqry wkey wval wout wup wdown wvoc te

def cal_target (n : i64) (seq : [16]i64) : [16][27]f32 =
  imap2 16 27 (\i j -> (if ((i < (n - 1)) && (seq[i + 1] == j)) then 1 else 0))

entry loss (dl : i64) (p : params) (seqs : [1][16]i64) (masks : [1][16][16]f32) : f32 =
  --  cal targets
   let targets = imap1 1 (\b -> cal_target dl seqs[b])
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   let te = (imap3 1 16 16 (\b m n -> wte[seqs[b][m]][n]))
   in nn64.loss masks wpe wqry wkey wval wout wup wdown wvoc te targets

def adam_opt_w [n] [m] (w : [n][m]f32) (mw : [n][m]f32) (vw : [n][m]f32)
  (dw : [n][m]f32) (step : i64) (lt_r : f32):
  ([n][m]f32, [n][m]f32, [n][m]f32) =
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

def adam_opt (p : params) (mp : params) (vp : params)
  (dp : params) (step : i64):
  (params,  params,  params) =
  let lt_r = 0.01 * (1 - (nn64.fromi64 step) / (nn64.fromi64 30000))
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

def grad_loss (dl : i64) (p : params) (seqs : [1][16]i64) (mask : [1][16][16]f32) :
        (
        [27][16]f32, -- dwte
        [16][16]f32, -- dwpe
        [16][16]f32, -- dwqry
        [16][16]f32, -- dwkey
        [16][16]f32, -- dwval
        [16][16]f32, -- dwout
        [64][16]f32, -- dwup
        [16][64]f32, -- dwdown
        [27][16]f32, -- dwvoc
        ) =
   let {wte, wpe, wqry, wkey, wval, wout, wup, wdown, wvoc} = p
   -- cal targets
  let targets = imap1 1 (\b -> cal_target dl seqs[b])
   -- cal voc embedding
  let te = (imap3 1 16 16 (\b m n -> wte[seqs[b][m]][n]))
   -- cal gradient
   let (dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc, dte) =
    nn64.grad_loss mask wpe wqry wkey wval wout wup wdown wvoc te targets
  -- DANGER: Not sure how to deal with batching here
  let dwte = (imap2 27 16 (\m n -> nn64.isum2 1 16 (\b k -> if (seqs[b][k] == m) then dte[b][k][n] else nn64.zero)))
   in  (dwte, dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc)

def cal_step (dl : i64) (p : params) (mp : params) (vp : params)
  (seqs : [1][16]i64) (mask : [1][16][16]f32)
  (step : i64) :
  (params,  params,  params) =
  -- cal gradient
  let (dwte, dwpe, dwqry, dwkey, dwval, dwout, dwup, dwdown, dwvoc) =
    grad_loss dl p seqs mask
  let dp = to_params dwte dwpe dwqry dwkey dwval dwout dwup dwdown dwvoc
  -- cal new model weights
  let (p', mp', vp') =
    adam_opt p mp vp dp step
  in (p', mp', vp')

entry train (p : params) (mp : params) (vp : params)
  (masks : [30000][1][16][16]f32) (dls : [30000]i64)
  (seqs : [30000][1][16]i64) =
  let (new_p, new_mp, new_vp) =
    loop (p', mp', vp') = (p, mp, vp)
    for step < 30000 do
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