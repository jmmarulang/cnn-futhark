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

module MAlonzo.Code.Replace where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.List
import qualified MAlonzo.Code.Agda.Builtin.Maybe
import qualified MAlonzo.Code.Agda.Builtin.Sigma
import qualified MAlonzo.Code.Ar
import qualified MAlonzo.Code.Data.Maybe.Base
import qualified MAlonzo.Code.Lang
import qualified MAlonzo.Code.LangEq
import qualified MAlonzo.Code.Relation.Nullary.Decidable.Core

-- Replace._.replace
d_replace_12 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 -> MAlonzo.Code.Lang.T_E_214
d_replace_12 v0 v1 v2 v3 v4 v5
  = let v6
          = MAlonzo.Code.LangEq.d__'8799''8305'__6 (coe v1) (coe v2) in
    coe
      (case coe v6 of
         MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v7 v8
           -> if coe v7
                then let v9
                           = seq
                               (coe v8)
                               (coe
                                  MAlonzo.Code.Data.Maybe.Base.du__'62''62''61'__72
                                  (coe
                                     MAlonzo.Code.LangEq.d__'8799''7497'__1798 (coe v0) (coe v1)
                                     (coe v3) (coe v4))
                                  (coe
                                     (\ v9 ->
                                        coe
                                          MAlonzo.Code.Agda.Builtin.Maybe.C_just_16
                                          (coe
                                             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 erased
                                             (coe v9))))) in
                     coe
                       (case coe v9 of
                          MAlonzo.Code.Agda.Builtin.Maybe.C_just_16 v10
                            -> coe seq (coe v10) (coe v5)
                          MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                            -> case coe v3 of
                                 MAlonzo.Code.Lang.C_var_216 v12
                                   -> coe MAlonzo.Code.Lang.C_var_216 v12
                                 MAlonzo.Code.Lang.C_zero_218 -> coe MAlonzo.Code.Lang.C_zero_218
                                 MAlonzo.Code.Lang.C_one_220 -> coe MAlonzo.Code.Lang.C_one_220
                                 MAlonzo.Code.Lang.C_imaps_222 v12
                                   -> case coe v1 of
                                        MAlonzo.Code.Lang.C_ar_34 v13
                                          -> coe
                                               MAlonzo.Code.Lang.C_imaps_222
                                               (d_replace_12
                                                  (coe
                                                     MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                                     (coe MAlonzo.Code.Lang.C_ix_32 (coe v13)))
                                                  (coe
                                                     MAlonzo.Code.Lang.C_ar_34
                                                     (coe MAlonzo.Code.Lang.d_unit_212))
                                                  (coe v2) (coe v12)
                                                  (coe
                                                     MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                                     (coe MAlonzo.Code.Lang.C_ix_32 (coe v13)) v4)
                                                  (coe
                                                     MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                                     (coe MAlonzo.Code.Lang.C_ix_32 (coe v13)) v5))
                                        _ -> MAlonzo.RTE.mazUnreachableError
                                 MAlonzo.Code.Lang.C_sels_224 v11 v12 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_sels_224 v11
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v11))
                                           (coe v2) (coe v12) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                           (coe v2) (coe v13) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_imap_226 v11 v12 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_imap_226 v11 v12
                                        (d_replace_12
                                           (coe
                                              MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                              (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)))
                                           (coe MAlonzo.Code.Lang.C_ar_34 (coe v12)) (coe v2)
                                           (coe v13)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)) v4)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)) v5))
                                 MAlonzo.Code.Lang.C_sel_228 v11 v13 v14
                                   -> case coe v1 of
                                        MAlonzo.Code.Lang.C_ar_34 v15
                                          -> coe
                                               MAlonzo.Code.Lang.C_sel_228 v11
                                               (d_replace_12
                                                  (coe v0)
                                                  (coe
                                                     MAlonzo.Code.Lang.C_ar_34
                                                     (coe
                                                        MAlonzo.Code.Ar.d__'8855'__54 () erased v11
                                                        v15))
                                                  (coe v2) (coe v13) (coe v4) (coe v5))
                                               (d_replace_12
                                                  (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                                  (coe v2) (coe v14) (coe v4) (coe v5))
                                        _ -> MAlonzo.RTE.mazUnreachableError
                                 MAlonzo.Code.Lang.C_imapb_230 v10 v11 v14 v15
                                   -> coe
                                        MAlonzo.Code.Lang.C_imapb_230 v10 v11 v14
                                        (d_replace_12
                                           (coe
                                              MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                              (coe MAlonzo.Code.Lang.C_ix_32 (coe v10)))
                                           (coe MAlonzo.Code.Lang.C_ar_34 (coe v11)) (coe v2)
                                           (coe v15)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_32 (coe v10)) v4)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_32 (coe v10)) v5))
                                 MAlonzo.Code.Lang.C_selb_232 v10 v12 v14 v15 v16
                                   -> coe
                                        MAlonzo.Code.Lang.C_selb_232 v10 v12 v14
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v12))
                                           (coe v2) (coe v15) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v10))
                                           (coe v2) (coe v16) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_sum_234 v11 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_sum_234 v11
                                        (d_replace_12
                                           (coe
                                              MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                              (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)))
                                           (coe v1) (coe v2) (coe v13)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)) v4)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)) v5))
                                 MAlonzo.Code.Lang.C_zero'45'but_236 v11 v13 v14 v15
                                   -> coe
                                        MAlonzo.Code.Lang.C_zero'45'but_236 v11
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                           (coe v2) (coe v13) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                           (coe v2) (coe v14) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe v1) (coe v2) (coe v15) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_slide_238 v11 v12 v13 v15 v16 v17 v18
                                   -> coe
                                        MAlonzo.Code.Lang.C_slide_238 v11 v12 v13
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                           (coe v2) (coe v15) (coe v4) (coe v5))
                                        v16
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v13))
                                           (coe v2) (coe v17) (coe v4) (coe v5))
                                        v18
                                 MAlonzo.Code.Lang.C_backslide_240 v11 v12 v13 v15 v16 v17 v18
                                   -> coe
                                        MAlonzo.Code.Lang.C_backslide_240 v11 v12 v13
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                           (coe v2) (coe v15) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v12))
                                           (coe v2) (coe v16) (coe v4) (coe v5))
                                        v17 v18
                                 MAlonzo.Code.Lang.C_bin_242 v12 v13 v14
                                   -> coe
                                        MAlonzo.Code.Lang.C_bin_242 v12
                                        (d_replace_12
                                           (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe v1) (coe v2) (coe v14) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_scaledown_244 v12 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_scaledown_244 v12
                                        (d_replace_12
                                           (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_let'8242'_246 v11 v13 v14
                                   -> coe
                                        MAlonzo.Code.Lang.C_let'8242'_246 v11
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v11))
                                           (coe v2) (coe v13) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe
                                              MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                              (coe MAlonzo.Code.Lang.C_ar_34 (coe v11)))
                                           (coe v1) (coe v2) (coe v14)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ar_34 (coe v11)) v4)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ar_34 (coe v11)) v5))
                                 MAlonzo.Code.Lang.C_un_248 v12 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_un_248 v12
                                        (d_replace_12
                                           (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                 _ -> MAlonzo.RTE.mazUnreachableError
                          _ -> MAlonzo.RTE.mazUnreachableError)
                else (let v9
                            = seq
                                (coe v8) (coe MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18) in
                      coe
                        (case coe v9 of
                           MAlonzo.Code.Agda.Builtin.Maybe.C_just_16 v10
                             -> coe seq (coe v10) (coe v5)
                           MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                             -> case coe v3 of
                                  MAlonzo.Code.Lang.C_var_216 v12
                                    -> coe MAlonzo.Code.Lang.C_var_216 v12
                                  MAlonzo.Code.Lang.C_zero_218 -> coe MAlonzo.Code.Lang.C_zero_218
                                  MAlonzo.Code.Lang.C_one_220 -> coe MAlonzo.Code.Lang.C_one_220
                                  MAlonzo.Code.Lang.C_imaps_222 v12
                                    -> case coe v1 of
                                         MAlonzo.Code.Lang.C_ar_34 v13
                                           -> coe
                                                MAlonzo.Code.Lang.C_imaps_222
                                                (d_replace_12
                                                   (coe
                                                      MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                                      (coe MAlonzo.Code.Lang.C_ix_32 (coe v13)))
                                                   (coe
                                                      MAlonzo.Code.Lang.C_ar_34
                                                      (coe MAlonzo.Code.Lang.d_unit_212))
                                                   (coe v2) (coe v12)
                                                   (coe
                                                      MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                                      (coe MAlonzo.Code.Lang.C_ix_32 (coe v13)) v4)
                                                   (coe
                                                      MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                                      (coe MAlonzo.Code.Lang.C_ix_32 (coe v13)) v5))
                                         _ -> MAlonzo.RTE.mazUnreachableError
                                  MAlonzo.Code.Lang.C_sels_224 v11 v12 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_sels_224 v11
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v11))
                                            (coe v2) (coe v12) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                            (coe v2) (coe v13) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_imap_226 v11 v12 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_imap_226 v11 v12
                                         (d_replace_12
                                            (coe
                                               MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)))
                                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v12)) (coe v2)
                                            (coe v13)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)) v4)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)) v5))
                                  MAlonzo.Code.Lang.C_sel_228 v11 v13 v14
                                    -> case coe v1 of
                                         MAlonzo.Code.Lang.C_ar_34 v15
                                           -> coe
                                                MAlonzo.Code.Lang.C_sel_228 v11
                                                (d_replace_12
                                                   (coe v0)
                                                   (coe
                                                      MAlonzo.Code.Lang.C_ar_34
                                                      (coe
                                                         MAlonzo.Code.Ar.d__'8855'__54 () erased v11
                                                         v15))
                                                   (coe v2) (coe v13) (coe v4) (coe v5))
                                                (d_replace_12
                                                   (coe v0)
                                                   (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                                   (coe v2) (coe v14) (coe v4) (coe v5))
                                         _ -> MAlonzo.RTE.mazUnreachableError
                                  MAlonzo.Code.Lang.C_imapb_230 v10 v11 v14 v15
                                    -> coe
                                         MAlonzo.Code.Lang.C_imapb_230 v10 v11 v14
                                         (d_replace_12
                                            (coe
                                               MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v10)))
                                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v11)) (coe v2)
                                            (coe v15)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v10)) v4)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v10)) v5))
                                  MAlonzo.Code.Lang.C_selb_232 v10 v12 v14 v15 v16
                                    -> coe
                                         MAlonzo.Code.Lang.C_selb_232 v10 v12 v14
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v12))
                                            (coe v2) (coe v15) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v10))
                                            (coe v2) (coe v16) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_sum_234 v11 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_sum_234 v11
                                         (d_replace_12
                                            (coe
                                               MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)))
                                            (coe v1) (coe v2) (coe v13)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)) v4)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v11)) v5))
                                  MAlonzo.Code.Lang.C_zero'45'but_236 v11 v13 v14 v15
                                    -> coe
                                         MAlonzo.Code.Lang.C_zero'45'but_236 v11
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                            (coe v2) (coe v13) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                            (coe v2) (coe v14) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v15) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_slide_238 v11 v12 v13 v15 v16 v17 v18
                                    -> coe
                                         MAlonzo.Code.Lang.C_slide_238 v11 v12 v13
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                            (coe v2) (coe v15) (coe v4) (coe v5))
                                         v16
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v13))
                                            (coe v2) (coe v17) (coe v4) (coe v5))
                                         v18
                                  MAlonzo.Code.Lang.C_backslide_240 v11 v12 v13 v15 v16 v17 v18
                                    -> coe
                                         MAlonzo.Code.Lang.C_backslide_240 v11 v12 v13
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v11))
                                            (coe v2) (coe v15) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v12))
                                            (coe v2) (coe v16) (coe v4) (coe v5))
                                         v17 v18
                                  MAlonzo.Code.Lang.C_bin_242 v12 v13 v14
                                    -> coe
                                         MAlonzo.Code.Lang.C_bin_242 v12
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v14) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_scaledown_244 v12 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_scaledown_244 v12
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_let'8242'_246 v11 v13 v14
                                    -> coe
                                         MAlonzo.Code.Lang.C_let'8242'_246 v11
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v11))
                                            (coe v2) (coe v13) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe
                                               MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                                               (coe MAlonzo.Code.Lang.C_ar_34 (coe v11)))
                                            (coe v1) (coe v2) (coe v14)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ar_34 (coe v11)) v4)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_512 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ar_34 (coe v11)) v5))
                                  MAlonzo.Code.Lang.C_un_248 v12 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_un_248 v12
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                  _ -> MAlonzo.RTE.mazUnreachableError
                           _ -> MAlonzo.RTE.mazUnreachableError))
         _ -> MAlonzo.RTE.mazUnreachableError)
-- Replace._.replace-let
d_replace'45'let_166 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> MAlonzo.Code.Lang.T_E_214
d_replace'45'let_166 v0 v1 v2
  = case coe v2 of
      MAlonzo.Code.Lang.C_var_216 v5
        -> coe MAlonzo.Code.Lang.C_var_216 v5
      MAlonzo.Code.Lang.C_zero_218 -> coe MAlonzo.Code.Lang.C_zero_218
      MAlonzo.Code.Lang.C_one_220 -> coe MAlonzo.Code.Lang.C_one_220
      MAlonzo.Code.Lang.C_imaps_222 v5
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v6
               -> coe
                    MAlonzo.Code.Lang.C_imaps_222
                    (d_replace'45'let_166
                       (coe
                          MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                          (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)))
                       (coe MAlonzo.Code.Lang.C_ar_34 (coe MAlonzo.Code.Lang.d_unit_212))
                       (coe v5))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sels_224 v4 v5 v6
        -> coe
             MAlonzo.Code.Lang.C_sels_224 v4
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v4)) (coe v5))
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) (coe v6))
      MAlonzo.Code.Lang.C_imap_226 v4 v5 v6
        -> coe
             MAlonzo.Code.Lang.C_imap_226 v4 v5
             (d_replace'45'let_166
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                   (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)))
                (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)) (coe v6))
      MAlonzo.Code.Lang.C_sel_228 v4 v6 v7
        -> case coe v1 of
             MAlonzo.Code.Lang.C_ar_34 v8
               -> coe
                    MAlonzo.Code.Lang.C_sel_228 v4
                    (d_replace'45'let_166
                       (coe v0)
                       (coe
                          MAlonzo.Code.Lang.C_ar_34
                          (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v4 v8))
                       (coe v6))
                    (d_replace'45'let_166
                       (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) (coe v7))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_imapb_230 v3 v4 v7 v8
        -> coe
             MAlonzo.Code.Lang.C_imapb_230 v3 v4 v7
             (d_replace'45'let_166
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                   (coe MAlonzo.Code.Lang.C_ix_32 (coe v3)))
                (coe MAlonzo.Code.Lang.C_ar_34 (coe v4)) (coe v8))
      MAlonzo.Code.Lang.C_selb_232 v3 v5 v7 v8 v9
        -> coe
             MAlonzo.Code.Lang.C_selb_232 v3 v5 v7
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)) (coe v8))
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v3)) (coe v9))
      MAlonzo.Code.Lang.C_sum_234 v4 v6
        -> coe
             MAlonzo.Code.Lang.C_sum_234 v4
             (d_replace'45'let_166
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                   (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)))
                (coe v1) (coe v6))
      MAlonzo.Code.Lang.C_zero'45'but_236 v4 v6 v7 v8
        -> coe
             MAlonzo.Code.Lang.C_zero'45'but_236 v4
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) (coe v6))
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) (coe v7))
             (d_replace'45'let_166 (coe v0) (coe v1) (coe v8))
      MAlonzo.Code.Lang.C_slide_238 v4 v5 v6 v8 v9 v10 v11
        -> coe
             MAlonzo.Code.Lang.C_slide_238 v4 v5 v6
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) (coe v8))
             v9
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe v10))
             v11
      MAlonzo.Code.Lang.C_backslide_240 v4 v5 v6 v8 v9 v10 v11
        -> coe
             MAlonzo.Code.Lang.C_backslide_240 v4 v5 v6
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) (coe v8))
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)) (coe v9))
             v10 v11
      MAlonzo.Code.Lang.C_bin_242 v5 v6 v7
        -> coe
             MAlonzo.Code.Lang.C_bin_242 v5
             (d_replace'45'let_166 (coe v0) (coe v1) (coe v6))
             (d_replace'45'let_166 (coe v0) (coe v1) (coe v7))
      MAlonzo.Code.Lang.C_scaledown_244 v5 v6
        -> coe
             MAlonzo.Code.Lang.C_scaledown_244 v5
             (d_replace'45'let_166 (coe v0) (coe v1) (coe v6))
      MAlonzo.Code.Lang.C_let'8242'_246 v4 v6 v7
        -> coe
             MAlonzo.Code.Lang.C_let'8242'_246 v4
             (d_replace'45'let_166
                (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v4)) (coe v6))
             (d_replace_12
                (coe
                   MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v4)))
                (coe v1) (coe MAlonzo.Code.Lang.C_ar_34 (coe v4))
                (coe
                   d_replace'45'let_166
                   (coe
                      MAlonzo.Code.Lang.C__'9657'__40 (coe v0)
                      (coe MAlonzo.Code.Lang.C_ar_34 (coe v4)))
                   (coe v1) (coe v7))
                (coe
                   MAlonzo.Code.Lang.d__'8593'_512 v0
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v4))
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v4))
                   (d_replace'45'let_166
                      (coe v0) (coe MAlonzo.Code.Lang.C_ar_34 (coe v4)) (coe v6)))
                (coe
                   MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60)))
      MAlonzo.Code.Lang.C_un_248 v5 v6
        -> coe
             MAlonzo.Code.Lang.C_un_248 v5
             (d_replace'45'let_166 (coe v0) (coe v1) (coe v6))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Replace.Test.ex₁
d_ex'8321'_238 :: MAlonzo.Code.Lang.T_E_214
d_ex'8321'_238
  = coe
      MAlonzo.Code.Lang.du_Lcon_1476
      (coe
         MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
         (coe
            MAlonzo.Code.Lang.C_ar_34
            (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
         (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
      (coe
         (\ v0 ->
            coe
              MAlonzo.Code.Lang.du_Let'45'syntax_1402
              (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)
              (coe
                 MAlonzo.Code.Lang.du_Let'45'syntax_1402
                 (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)
                 (coe
                    MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_plus_190)
                    (coe
                       v0
                       (MAlonzo.Code.Lang.d_ext_1412
                          (coe MAlonzo.Code.Lang.C_ε_38)
                          (coe
                             MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                             (coe
                                MAlonzo.Code.Lang.C_ar_34
                                (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                             (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                       (coe MAlonzo.Code.Lang.C_zero_1334))
                    (coe
                       v0
                       (MAlonzo.Code.Lang.d_ext_1412
                          (coe MAlonzo.Code.Lang.C_ε_38)
                          (coe
                             MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                             (coe
                                MAlonzo.Code.Lang.C_ar_34
                                (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                             (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                       (coe MAlonzo.Code.Lang.C_zero_1334)))
                 (coe
                    (\ v1 ->
                       coe
                         MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_plus_190)
                         (coe
                            v1
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__40
                               (coe
                                  MAlonzo.Code.Lang.d_ext_1412 (coe MAlonzo.Code.Lang.C_ε_38)
                                  (coe
                                     MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                                     (coe
                                        MAlonzo.Code.Lang.C_ar_34
                                        (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                                     (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                               (coe
                                  MAlonzo.Code.Lang.C_ar_34
                                  (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                            (coe MAlonzo.Code.Lang.C_zero_1334))
                         (coe
                            v1
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__40
                               (coe
                                  MAlonzo.Code.Lang.d_ext_1412 (coe MAlonzo.Code.Lang.C_ε_38)
                                  (coe
                                     MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                                     (coe
                                        MAlonzo.Code.Lang.C_ar_34
                                        (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                                     (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                               (coe
                                  MAlonzo.Code.Lang.C_ar_34
                                  (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                            (coe MAlonzo.Code.Lang.C_zero_1334)))))
              (coe
                 (\ v1 ->
                    coe
                      v1
                      (coe
                         MAlonzo.Code.Lang.C__'9657'__40
                         (coe
                            MAlonzo.Code.Lang.d_ext_1412 (coe MAlonzo.Code.Lang.C_ε_38)
                            (coe
                               MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                               (coe
                                  MAlonzo.Code.Lang.C_ar_34
                                  (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                               (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                         (coe
                            MAlonzo.Code.Lang.C_ar_34
                            (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                      (coe MAlonzo.Code.Lang.C_zero_1334)))))
-- Replace.Test.ex-repl
d_ex'45'repl_246 :: MAlonzo.Code.Lang.T_E_214
d_ex'45'repl_246
  = coe
      d_replace_12
      (coe
         MAlonzo.Code.Lang.d_ext_1412 (coe MAlonzo.Code.Lang.C_ε_38)
         (coe
            MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
            (coe
               MAlonzo.Code.Lang.C_ar_34
               (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
            (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
      (coe
         MAlonzo.Code.Lang.C_ar_34
         (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
      (coe
         MAlonzo.Code.Lang.C_ar_34
         (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
      (coe d_ex'8321'_238)
      (coe
         MAlonzo.Code.Lang.C_bin_242 (coe MAlonzo.Code.Lang.C_plus_190)
         (coe MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60))
         (coe
            MAlonzo.Code.Lang.C_var_216 (coe MAlonzo.Code.Lang.C_here_60)))
      (coe MAlonzo.Code.Lang.C_one_220)
