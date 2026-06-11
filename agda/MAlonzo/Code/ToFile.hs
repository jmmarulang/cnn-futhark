{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE EmptyCase #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -Wno-overlapping-patterns #-}

module MAlonzo.Code.ToFile where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.IO
import qualified MAlonzo.Code.Agda.Builtin.List
import qualified MAlonzo.Code.Agda.Builtin.String
import qualified MAlonzo.Code.Ar
import qualified MAlonzo.Code.Extraction
import qualified MAlonzo.Code.IO.Base
import qualified MAlonzo.Code.IO.Finite
import qualified MAlonzo.Code.Lang
import qualified MAlonzo.Code.Level

-- ToFile.grad-mgpt-loss-s
d_grad'45'mgpt'45'loss'45's_2 ::
  MAlonzo.Code.Agda.Builtin.String.T_String_6
d_grad'45'mgpt'45'loss'45's_2
  = coe
      MAlonzo.Code.Extraction.d_pp_300
      (coe
         MAlonzo.Code.Lang.d_ext_1208 (coe MAlonzo.Code.Lang.C_ε_14)
         (coe
            MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
            (coe
               MAlonzo.Code.Lang.C_ar_10
               (coe
                  MAlonzo.Code.Ar.d__'8855'__54 () erased MAlonzo.Code.Lang.d_SL_2320
                  MAlonzo.Code.Lang.d_SL_2320))
            (coe
               MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
               (coe
                  MAlonzo.Code.Lang.C_ar_10
                  (coe
                     MAlonzo.Code.Ar.d__'8855'__54 () erased MAlonzo.Code.Lang.d_SL_2320
                     MAlonzo.Code.Lang.d_ED_2314))
               (coe
                  MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                  (coe
                     MAlonzo.Code.Lang.C_ar_10
                     (coe
                        MAlonzo.Code.Ar.d__'8855'__54 () erased MAlonzo.Code.Lang.d_ED_2314
                        MAlonzo.Code.Lang.d_ED_2314))
                  (coe
                     MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                     (coe
                        MAlonzo.Code.Lang.C_ar_10
                        (coe
                           MAlonzo.Code.Ar.d__'8855'__54 () erased MAlonzo.Code.Lang.d_ED_2314
                           MAlonzo.Code.Lang.d_ED_2314))
                     (coe
                        MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                        (coe
                           MAlonzo.Code.Lang.C_ar_10
                           (coe
                              MAlonzo.Code.Ar.d__'8855'__54 () erased MAlonzo.Code.Lang.d_ED_2314
                              MAlonzo.Code.Lang.d_ED_2314))
                        (coe
                           MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                           (coe
                              MAlonzo.Code.Lang.C_ar_10
                              (coe
                                 MAlonzo.Code.Ar.d__'8855'__54 () erased MAlonzo.Code.Lang.d_ED_2314
                                 MAlonzo.Code.Lang.d_ED_2314))
                           (coe
                              MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                              (coe
                                 MAlonzo.Code.Lang.C_ar_10
                                 (coe
                                    MAlonzo.Code.Ar.d__'8855'__54 () erased
                                    (coe
                                       MAlonzo.Code.Ar.d__'8855'__54 () erased
                                       MAlonzo.Code.Lang.d_FD_2322 MAlonzo.Code.Lang.d_ED_2314)
                                    MAlonzo.Code.Lang.d_ED_2314))
                              (coe
                                 MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                                 (coe
                                    MAlonzo.Code.Lang.C_ar_10
                                    (coe
                                       MAlonzo.Code.Ar.d__'8855'__54 () erased
                                       MAlonzo.Code.Lang.d_ED_2314
                                       (coe
                                          MAlonzo.Code.Ar.d__'8855'__54 () erased
                                          MAlonzo.Code.Lang.d_FD_2322 MAlonzo.Code.Lang.d_ED_2314)))
                                 (coe
                                    MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                                    (coe
                                       MAlonzo.Code.Lang.C_ar_10
                                       (coe
                                          MAlonzo.Code.Ar.d__'8855'__54 () erased
                                          MAlonzo.Code.Lang.d_VO_2326 MAlonzo.Code.Lang.d_ED_2314))
                                    (coe
                                       MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                                       (coe
                                          MAlonzo.Code.Lang.C_ar_10
                                          (coe
                                             MAlonzo.Code.Ar.d__'8855'__54 () erased
                                             MAlonzo.Code.Lang.d_SL_2320
                                             MAlonzo.Code.Lang.d_ED_2314))
                                       (coe
                                          MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                                          (coe
                                             MAlonzo.Code.Lang.C_ar_10
                                             (coe
                                                MAlonzo.Code.Ar.d__'8855'__54 () erased
                                                MAlonzo.Code.Lang.d_SL_2320
                                                MAlonzo.Code.Lang.d_VO_2326))
                                          (coe
                                             MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))))))))))))
      (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)
      (coe MAlonzo.Code.Lang.d_mgpt'45'loss'45'e_2330)
      (coe
         MAlonzo.Code.Extraction.C__'9657'__230
         (coe
            MAlonzo.Code.Extraction.C__'9657'__230
            (coe
               MAlonzo.Code.Extraction.C__'9657'__230
               (coe
                  MAlonzo.Code.Extraction.C__'9657'__230
                  (coe
                     MAlonzo.Code.Extraction.C__'9657'__230
                     (coe
                        MAlonzo.Code.Extraction.C__'9657'__230
                        (coe
                           MAlonzo.Code.Extraction.C__'9657'__230
                           (coe
                              MAlonzo.Code.Extraction.C__'9657'__230
                              (coe
                                 MAlonzo.Code.Extraction.C__'9657'__230
                                 (coe
                                    MAlonzo.Code.Extraction.C__'9657'__230
                                    (coe
                                       MAlonzo.Code.Extraction.C__'9657'__230
                                       (coe MAlonzo.Code.Extraction.C_ε_228)
                                       ("mask" :: Data.Text.Text))
                                    ("wpe" :: Data.Text.Text))
                                 ("wqry" :: Data.Text.Text))
                              ("wkey" :: Data.Text.Text))
                           ("wval" :: Data.Text.Text))
                        ("wout" :: Data.Text.Text))
                     ("wup" :: Data.Text.Text))
                  ("wdown" :: Data.Text.Text))
               ("wvoc" :: Data.Text.Text))
            ("wseq" :: Data.Text.Text))
         ("target" :: Data.Text.Text))
main = coe d_main_4
-- ToFile.main
d_main_4 ::
  MAlonzo.Code.Agda.Builtin.IO.T_IO_8
    AgdaAny MAlonzo.Code.Level.T_Lift_8
d_main_4
  = coe
      MAlonzo.Code.IO.Base.du_run_122 (coe MAlonzo.Code.Level.d_0ℓ_22)
      (coe
         MAlonzo.Code.IO.Finite.d_putStrLn_28
         (coe MAlonzo.Code.Level.d_0ℓ_22)
         (coe d_grad'45'mgpt'45'loss'45's_2))
