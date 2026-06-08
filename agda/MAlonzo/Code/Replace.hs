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
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_E_182 ->
  MAlonzo.Code.Lang.T_E_182 ->
  MAlonzo.Code.Lang.T_E_182 -> MAlonzo.Code.Lang.T_E_182
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
                                     MAlonzo.Code.LangEq.d__'8799''7497'__1756 (coe v0) (coe v1)
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
                                 MAlonzo.Code.Lang.C_var_184 v12
                                   -> coe MAlonzo.Code.Lang.C_var_184 v12
                                 MAlonzo.Code.Lang.C_zero_186 -> coe MAlonzo.Code.Lang.C_zero_186
                                 MAlonzo.Code.Lang.C_one_188 -> coe MAlonzo.Code.Lang.C_one_188
                                 MAlonzo.Code.Lang.C_imaps_190 v12
                                   -> case coe v1 of
                                        MAlonzo.Code.Lang.C_ar_10 v13
                                          -> coe
                                               MAlonzo.Code.Lang.C_imaps_190
                                               (d_replace_12
                                                  (coe
                                                     MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                                     (coe MAlonzo.Code.Lang.C_ix_8 (coe v13)))
                                                  (coe
                                                     MAlonzo.Code.Lang.C_ar_10
                                                     (coe MAlonzo.Code.Lang.d_unit_180))
                                                  (coe v2) (coe v12)
                                                  (coe
                                                     MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                                     (coe MAlonzo.Code.Lang.C_ix_8 (coe v13)) v4)
                                                  (coe
                                                     MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                                     (coe MAlonzo.Code.Lang.C_ix_8 (coe v13)) v5))
                                        _ -> MAlonzo.RTE.mazUnreachableError
                                 MAlonzo.Code.Lang.C_sels_192 v11 v12 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_sels_192 v11
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v11))
                                           (coe v2) (coe v12) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                           (coe v2) (coe v13) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_imap_194 v11 v12 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_imap_194 v11 v12
                                        (d_replace_12
                                           (coe
                                              MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                              (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)))
                                           (coe MAlonzo.Code.Lang.C_ar_10 (coe v12)) (coe v2)
                                           (coe v13)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)) v4)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)) v5))
                                 MAlonzo.Code.Lang.C_sel_196 v11 v13 v14
                                   -> case coe v1 of
                                        MAlonzo.Code.Lang.C_ar_10 v15
                                          -> coe
                                               MAlonzo.Code.Lang.C_sel_196 v11
                                               (d_replace_12
                                                  (coe v0)
                                                  (coe
                                                     MAlonzo.Code.Lang.C_ar_10
                                                     (coe
                                                        MAlonzo.Code.Ar.d__'8855'__54 () erased v11
                                                        v15))
                                                  (coe v2) (coe v13) (coe v4) (coe v5))
                                               (d_replace_12
                                                  (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                                  (coe v2) (coe v14) (coe v4) (coe v5))
                                        _ -> MAlonzo.RTE.mazUnreachableError
                                 MAlonzo.Code.Lang.C_imapb_198 v10 v11 v14 v15
                                   -> coe
                                        MAlonzo.Code.Lang.C_imapb_198 v10 v11 v14
                                        (d_replace_12
                                           (coe
                                              MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                              (coe MAlonzo.Code.Lang.C_ix_8 (coe v10)))
                                           (coe MAlonzo.Code.Lang.C_ar_10 (coe v11)) (coe v2)
                                           (coe v15)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_8 (coe v10)) v4)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_8 (coe v10)) v5))
                                 MAlonzo.Code.Lang.C_selb_200 v10 v12 v14 v15 v16
                                   -> coe
                                        MAlonzo.Code.Lang.C_selb_200 v10 v12 v14
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v12))
                                           (coe v2) (coe v15) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v10))
                                           (coe v2) (coe v16) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_sum_202 v11 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_sum_202 v11
                                        (d_replace_12
                                           (coe
                                              MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                              (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)))
                                           (coe v1) (coe v2) (coe v13)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)) v4)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)) v5))
                                 MAlonzo.Code.Lang.C_zero'45'but_204 v11 v13 v14 v15
                                   -> coe
                                        MAlonzo.Code.Lang.C_zero'45'but_204 v11
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                           (coe v2) (coe v13) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                           (coe v2) (coe v14) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe v1) (coe v2) (coe v15) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_slide_206 v11 v12 v13 v15 v16 v17 v18
                                   -> coe
                                        MAlonzo.Code.Lang.C_slide_206 v11 v12 v13
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                           (coe v2) (coe v15) (coe v4) (coe v5))
                                        v16
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v13))
                                           (coe v2) (coe v17) (coe v4) (coe v5))
                                        v18
                                 MAlonzo.Code.Lang.C_backslide_208 v11 v12 v13 v15 v16 v17 v18
                                   -> coe
                                        MAlonzo.Code.Lang.C_backslide_208 v11 v12 v13
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                           (coe v2) (coe v15) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v12))
                                           (coe v2) (coe v16) (coe v4) (coe v5))
                                        v17 v18
                                 MAlonzo.Code.Lang.C_bin_210 v12 v13 v14
                                   -> coe
                                        MAlonzo.Code.Lang.C_bin_210 v12
                                        (d_replace_12
                                           (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe v0) (coe v1) (coe v2) (coe v14) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_scaledown_212 v12 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_scaledown_212 v12
                                        (d_replace_12
                                           (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                 MAlonzo.Code.Lang.C_let'8242'_214 v11 v13 v14
                                   -> coe
                                        MAlonzo.Code.Lang.C_let'8242'_214 v11
                                        (d_replace_12
                                           (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v11))
                                           (coe v2) (coe v13) (coe v4) (coe v5))
                                        (d_replace_12
                                           (coe
                                              MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                              (coe MAlonzo.Code.Lang.C_ar_10 (coe v11)))
                                           (coe v1) (coe v2) (coe v14)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ar_10 (coe v11)) v4)
                                           (coe
                                              MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                              (coe MAlonzo.Code.Lang.C_ar_10 (coe v11)) v5))
                                 MAlonzo.Code.Lang.C_un_216 v12 v13
                                   -> coe
                                        MAlonzo.Code.Lang.C_un_216 v12
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
                                  MAlonzo.Code.Lang.C_var_184 v12
                                    -> coe MAlonzo.Code.Lang.C_var_184 v12
                                  MAlonzo.Code.Lang.C_zero_186 -> coe MAlonzo.Code.Lang.C_zero_186
                                  MAlonzo.Code.Lang.C_one_188 -> coe MAlonzo.Code.Lang.C_one_188
                                  MAlonzo.Code.Lang.C_imaps_190 v12
                                    -> case coe v1 of
                                         MAlonzo.Code.Lang.C_ar_10 v13
                                           -> coe
                                                MAlonzo.Code.Lang.C_imaps_190
                                                (d_replace_12
                                                   (coe
                                                      MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                                      (coe MAlonzo.Code.Lang.C_ix_8 (coe v13)))
                                                   (coe
                                                      MAlonzo.Code.Lang.C_ar_10
                                                      (coe MAlonzo.Code.Lang.d_unit_180))
                                                   (coe v2) (coe v12)
                                                   (coe
                                                      MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                                      (coe MAlonzo.Code.Lang.C_ix_8 (coe v13)) v4)
                                                   (coe
                                                      MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                                      (coe MAlonzo.Code.Lang.C_ix_8 (coe v13)) v5))
                                         _ -> MAlonzo.RTE.mazUnreachableError
                                  MAlonzo.Code.Lang.C_sels_192 v11 v12 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_sels_192 v11
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v11))
                                            (coe v2) (coe v12) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                            (coe v2) (coe v13) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_imap_194 v11 v12 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_imap_194 v11 v12
                                         (d_replace_12
                                            (coe
                                               MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)))
                                            (coe MAlonzo.Code.Lang.C_ar_10 (coe v12)) (coe v2)
                                            (coe v13)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)) v4)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)) v5))
                                  MAlonzo.Code.Lang.C_sel_196 v11 v13 v14
                                    -> case coe v1 of
                                         MAlonzo.Code.Lang.C_ar_10 v15
                                           -> coe
                                                MAlonzo.Code.Lang.C_sel_196 v11
                                                (d_replace_12
                                                   (coe v0)
                                                   (coe
                                                      MAlonzo.Code.Lang.C_ar_10
                                                      (coe
                                                         MAlonzo.Code.Ar.d__'8855'__54 () erased v11
                                                         v15))
                                                   (coe v2) (coe v13) (coe v4) (coe v5))
                                                (d_replace_12
                                                   (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                                   (coe v2) (coe v14) (coe v4) (coe v5))
                                         _ -> MAlonzo.RTE.mazUnreachableError
                                  MAlonzo.Code.Lang.C_imapb_198 v10 v11 v14 v15
                                    -> coe
                                         MAlonzo.Code.Lang.C_imapb_198 v10 v11 v14
                                         (d_replace_12
                                            (coe
                                               MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v10)))
                                            (coe MAlonzo.Code.Lang.C_ar_10 (coe v11)) (coe v2)
                                            (coe v15)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v10)) v4)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v10)) v5))
                                  MAlonzo.Code.Lang.C_selb_200 v10 v12 v14 v15 v16
                                    -> coe
                                         MAlonzo.Code.Lang.C_selb_200 v10 v12 v14
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v12))
                                            (coe v2) (coe v15) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v10))
                                            (coe v2) (coe v16) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_sum_202 v11 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_sum_202 v11
                                         (d_replace_12
                                            (coe
                                               MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)))
                                            (coe v1) (coe v2) (coe v13)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)) v4)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v11)) v5))
                                  MAlonzo.Code.Lang.C_zero'45'but_204 v11 v13 v14 v15
                                    -> coe
                                         MAlonzo.Code.Lang.C_zero'45'but_204 v11
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                            (coe v2) (coe v13) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                            (coe v2) (coe v14) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v15) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_slide_206 v11 v12 v13 v15 v16 v17 v18
                                    -> coe
                                         MAlonzo.Code.Lang.C_slide_206 v11 v12 v13
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                            (coe v2) (coe v15) (coe v4) (coe v5))
                                         v16
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v13))
                                            (coe v2) (coe v17) (coe v4) (coe v5))
                                         v18
                                  MAlonzo.Code.Lang.C_backslide_208 v11 v12 v13 v15 v16 v17 v18
                                    -> coe
                                         MAlonzo.Code.Lang.C_backslide_208 v11 v12 v13
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ix_8 (coe v11))
                                            (coe v2) (coe v15) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v12))
                                            (coe v2) (coe v16) (coe v4) (coe v5))
                                         v17 v18
                                  MAlonzo.Code.Lang.C_bin_210 v12 v13 v14
                                    -> coe
                                         MAlonzo.Code.Lang.C_bin_210 v12
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v14) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_scaledown_212 v12 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_scaledown_212 v12
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                  MAlonzo.Code.Lang.C_let'8242'_214 v11 v13 v14
                                    -> coe
                                         MAlonzo.Code.Lang.C_let'8242'_214 v11
                                         (d_replace_12
                                            (coe v0) (coe MAlonzo.Code.Lang.C_ar_10 (coe v11))
                                            (coe v2) (coe v13) (coe v4) (coe v5))
                                         (d_replace_12
                                            (coe
                                               MAlonzo.Code.Lang.C__'9657'__16 (coe v0)
                                               (coe MAlonzo.Code.Lang.C_ar_10 (coe v11)))
                                            (coe v1) (coe v2) (coe v14)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ar_10 (coe v11)) v4)
                                            (coe
                                               MAlonzo.Code.Lang.d__'8593'_448 v0 v2
                                               (coe MAlonzo.Code.Lang.C_ar_10 (coe v11)) v5))
                                  MAlonzo.Code.Lang.C_un_216 v12 v13
                                    -> coe
                                         MAlonzo.Code.Lang.C_un_216 v12
                                         (d_replace_12
                                            (coe v0) (coe v1) (coe v2) (coe v13) (coe v4) (coe v5))
                                  _ -> MAlonzo.RTE.mazUnreachableError
                           _ -> MAlonzo.RTE.mazUnreachableError))
         _ -> MAlonzo.RTE.mazUnreachableError)
-- Replace.Test.ex₁
d_ex'8321'_166 :: MAlonzo.Code.Lang.T_E_182
d_ex'8321'_166
  = coe
      MAlonzo.Code.Lang.du_Lcon_1272
      (coe
         MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
         (coe
            MAlonzo.Code.Lang.C_ar_10
            (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
         (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
      (coe
         (\ v0 ->
            coe
              MAlonzo.Code.Lang.du_Let'45'syntax_1198
              (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)
              (coe
                 MAlonzo.Code.Lang.du_Let'45'syntax_1198
                 (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)
                 (coe
                    MAlonzo.Code.Lang.C_bin_210 (coe MAlonzo.Code.Lang.C_plus_158)
                    (coe
                       v0
                       (MAlonzo.Code.Lang.d_ext_1208
                          (coe MAlonzo.Code.Lang.C_ε_14)
                          (coe
                             MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                             (coe
                                MAlonzo.Code.Lang.C_ar_10
                                (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                             (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                       (coe MAlonzo.Code.Lang.C_zero_1130))
                    (coe
                       v0
                       (MAlonzo.Code.Lang.d_ext_1208
                          (coe MAlonzo.Code.Lang.C_ε_14)
                          (coe
                             MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                             (coe
                                MAlonzo.Code.Lang.C_ar_10
                                (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                             (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                       (coe MAlonzo.Code.Lang.C_zero_1130)))
                 (coe
                    (\ v1 ->
                       coe
                         MAlonzo.Code.Lang.C_bin_210 (coe MAlonzo.Code.Lang.C_plus_158)
                         (coe
                            v1
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__16
                               (coe
                                  MAlonzo.Code.Lang.d_ext_1208 (coe MAlonzo.Code.Lang.C_ε_14)
                                  (coe
                                     MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                                     (coe
                                        MAlonzo.Code.Lang.C_ar_10
                                        (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                                     (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                               (coe
                                  MAlonzo.Code.Lang.C_ar_10
                                  (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                            (coe MAlonzo.Code.Lang.C_zero_1130))
                         (coe
                            v1
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__16
                               (coe
                                  MAlonzo.Code.Lang.d_ext_1208 (coe MAlonzo.Code.Lang.C_ε_14)
                                  (coe
                                     MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                                     (coe
                                        MAlonzo.Code.Lang.C_ar_10
                                        (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                                     (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                               (coe
                                  MAlonzo.Code.Lang.C_ar_10
                                  (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                            (coe MAlonzo.Code.Lang.C_zero_1130)))))
              (coe
                 (\ v1 ->
                    coe
                      v1
                      (coe
                         MAlonzo.Code.Lang.C__'9657'__16
                         (coe
                            MAlonzo.Code.Lang.d_ext_1208 (coe MAlonzo.Code.Lang.C_ε_14)
                            (coe
                               MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
                               (coe
                                  MAlonzo.Code.Lang.C_ar_10
                                  (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                               (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                         (coe
                            MAlonzo.Code.Lang.C_ar_10
                            (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
                      (coe MAlonzo.Code.Lang.C_zero_1130)))))
-- Replace.Test.ex-repl
d_ex'45'repl_174 :: MAlonzo.Code.Lang.T_E_182
d_ex'45'repl_174
  = coe
      d_replace_12
      (coe
         MAlonzo.Code.Lang.d_ext_1208 (coe MAlonzo.Code.Lang.C_ε_14)
         (coe
            MAlonzo.Code.Agda.Builtin.List.C__'8759'__22
            (coe
               MAlonzo.Code.Lang.C_ar_10
               (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
            (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16)))
      (coe
         MAlonzo.Code.Lang.C_ar_10
         (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
      (coe
         MAlonzo.Code.Lang.C_ar_10
         (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
      (coe d_ex'8321'_166)
      (coe
         MAlonzo.Code.Lang.C_bin_210 (coe MAlonzo.Code.Lang.C_plus_158)
         (coe MAlonzo.Code.Lang.C_var_184 (coe MAlonzo.Code.Lang.C_here_36))
         (coe
            MAlonzo.Code.Lang.C_var_184 (coe MAlonzo.Code.Lang.C_here_36)))
      (coe MAlonzo.Code.Lang.C_one_188)
