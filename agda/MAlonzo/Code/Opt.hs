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

module MAlonzo.Code.Opt where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Bool
import qualified MAlonzo.Code.Agda.Builtin.Equality
import qualified MAlonzo.Code.Agda.Builtin.List
import qualified MAlonzo.Code.Agda.Builtin.Maybe
import qualified MAlonzo.Code.Agda.Builtin.Sigma
import qualified MAlonzo.Code.Ar
import qualified MAlonzo.Code.Data.Irrelevant
import qualified MAlonzo.Code.Data.List.Relation.Unary.All
import qualified MAlonzo.Code.Data.Nat.Properties
import qualified MAlonzo.Code.Eval
import qualified MAlonzo.Code.Lang
import qualified MAlonzo.Code.LangEq
import qualified MAlonzo.Code.Real
import qualified MAlonzo.Code.Relation.Nullary.Decidable.Core
import qualified MAlonzo.Code.Relation.Nullary.Reflects

-- Opt._.fromℕ
d_fromℕ_30 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 -> Integer -> AgdaAny
d_fromℕ_30 v0 ~v1 = du_fromℕ_30 v0
du_fromℕ_30 :: MAlonzo.Code.Real.T_Real_2 -> Integer -> AgdaAny
du_fromℕ_30 v0 = coe MAlonzo.Code.Real.d_fromℕ_30 (coe v0)
-- Opt._._≈ᵉ_
d__'8776''7497'__72 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> MAlonzo.Code.Lang.T_E_214 -> ()
d__'8776''7497'__72 = erased
-- Opt._._≈ᶜ_
d__'8776''7580'__74 a0 a1 a2 a3 a4 = ()
-- Opt._.eval
d_eval_78 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> AgdaAny -> AgdaAny
d_eval_78 v0 ~v1 = du_eval_78 v0
du_eval_78 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> AgdaAny -> AgdaAny
du_eval_78 v0 = coe MAlonzo.Code.Eval.d_eval_114 (coe v0)
-- Opt._.⟦_⟧ᶜ
d_'10214'_'10215''7580'_128 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 -> MAlonzo.Code.Lang.T_Ctx_36 -> ()
d_'10214'_'10215''7580'_128 = erased
-- Opt.∷-inj₂
d_'8759''45'inj'8322'_198 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'8759''45'inj'8322'_198 = erased
-- Opt.++-inj₂
d_'43''43''45'inj'8322'_200 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_'43''43''45'inj'8322'_200 = erased
-- Opt.opt
d_opt_214 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14
d_opt_214 v0 ~v1 v2 v3 v4 = du_opt_214 v0 v2 v3 v4
du_opt_214 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14
du_opt_214 v0 v1 v2 v3
  = case coe v3 of
      MAlonzo.Code.Lang.C_var_216 v6
        -> coe
             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
             (coe MAlonzo.Code.Lang.C_var_216 v6) erased
      MAlonzo.Code.Lang.C_zero_218
        -> coe
             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
             (coe MAlonzo.Code.Lang.C_zero_218) erased
      MAlonzo.Code.Lang.C_one_220
        -> coe
             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
             (coe MAlonzo.Code.Lang.C_one_220) erased
      MAlonzo.Code.Lang.C_imaps_222 v6
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v7
               -> let v8
                        = coe
                            du_opt_214 (coe v0)
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v7)))
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe MAlonzo.Code.Lang.d_unit_212))
                            (coe v6) in
                  coe
                    (case coe v8 of
                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v9 v10
                         -> coe
                              MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                              (coe MAlonzo.Code.Lang.C_imaps_222 v9)
                              (coe
                                 (\ v11 v12 ->
                                    coe
                                      v10
                                      (coe
                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v11)
                                         (coe v12))
                                      (coe
                                         MAlonzo.Code.Data.List.Relation.Unary.All.C_'91''93'_50)))
                       _ -> MAlonzo.RTE.mazUnreachableError)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sels_224 v5 v6 v7
        -> let v8
                 = coe
                     du_opt_214 (coe v0) (coe v1)
                     (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)) (coe v6) in
           coe
             (let v9
                    = coe
                        du_opt_214 (coe v0) (coe v1)
                        (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) (coe v7) in
              coe
                (case coe v8 of
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v10 v11
                     -> let v12
                              = case coe v9 of
                                  MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                    -> coe
                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                         (coe MAlonzo.Code.Lang.C_sels_224 v5 v10 v12) erased
                                  _ -> MAlonzo.RTE.mazUnreachableError in
                        coe
                          (case coe v10 of
                             MAlonzo.Code.Lang.C_zero_218
                               -> coe
                                    seq (coe v9)
                                    (coe
                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                       (coe MAlonzo.Code.Lang.C_zero_218)
                                       (coe
                                          (\ v15 v16 ->
                                             coe
                                               v11 v15
                                               (MAlonzo.Code.Eval.d_eval_114
                                                  (coe v0) (coe v1)
                                                  (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) (coe v7)
                                                  (coe v15)))))
                             MAlonzo.Code.Lang.C_one_220
                               -> coe
                                    seq (coe v9)
                                    (coe
                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                       (coe MAlonzo.Code.Lang.C_one_220)
                                       (coe
                                          (\ v15 v16 ->
                                             coe
                                               v11 v15
                                               (MAlonzo.Code.Eval.d_eval_114
                                                  (coe v0) (coe v1)
                                                  (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) (coe v7)
                                                  (coe v15)))))
                             MAlonzo.Code.Lang.C_imaps_222 v15
                               -> case coe v9 of
                                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                                      -> coe
                                           MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                           (coe
                                              MAlonzo.Code.Lang.d_sub_554
                                              (coe
                                                 MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                 (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                              (coe
                                                 MAlonzo.Code.Lang.C_ar_34
                                                 (coe MAlonzo.Code.Agda.Builtin.List.C_'91''93'_16))
                                              (coe v1) (coe v15)
                                              (coe
                                                 MAlonzo.Code.Lang.C__'9657'__516
                                                 (MAlonzo.Code.Lang.d_sub'45'id_548 (coe v1)) v16))
                                           erased
                                    _ -> MAlonzo.RTE.mazUnreachableError
                             MAlonzo.Code.Lang.C_sum_234 v14 v16
                               -> case coe v9 of
                                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v17 v18
                                      -> coe
                                           MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                           (coe
                                              MAlonzo.Code.Lang.C_sum_234 v14
                                              (coe
                                                 MAlonzo.Code.Lang.C_sels_224 v5 v16
                                                 (coe
                                                    MAlonzo.Code.Lang.d__'8593'_500 v1
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v5))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v14)) v17)))
                                           erased
                                    _ -> MAlonzo.RTE.mazUnreachableError
                             MAlonzo.Code.Lang.C_zero'45'but_236 v14 v16 v17 v18
                               -> case coe v9 of
                                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v19 v20
                                      -> coe
                                           MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                           (coe
                                              MAlonzo.Code.Lang.C_zero'45'but_236 v14 v16 v17
                                              (coe MAlonzo.Code.Lang.C_sels_224 v5 v18 v19))
                                           erased
                                    _ -> MAlonzo.RTE.mazUnreachableError
                             MAlonzo.Code.Lang.C_bin_242 v15 v16 v17
                               -> coe
                                    seq (coe v15)
                                    (case coe v9 of
                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v18 v19
                                         -> coe
                                              MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                              (coe
                                                 MAlonzo.Code.Lang.C_bin_242 v15
                                                 (coe MAlonzo.Code.Lang.C_sels_224 v5 v16 v18)
                                                 (coe MAlonzo.Code.Lang.C_sels_224 v5 v17 v18))
                                              erased
                                       _ -> MAlonzo.RTE.mazUnreachableError)
                             _ -> coe v12)
                   _ -> MAlonzo.RTE.mazUnreachableError))
      MAlonzo.Code.Lang.C_imap_226 v5 v6 v7
        -> let v8
                 = coe
                     du_opt_214 (coe v0)
                     (coe
                        MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                        (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                     (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe v7) in
           coe
             (case coe v8 of
                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v9 v10
                  -> let v11 = coe MAlonzo.Code.LangEq.du_isLet_1704 (coe v9) in
                     coe
                       (case coe v11 of
                          MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v12 v13
                            -> let v14
                                     = coe
                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                         (coe MAlonzo.Code.Lang.C_imap_226 v5 v6 v9)
                                         (coe
                                            (\ v14 v15 ->
                                               coe
                                                 v10
                                                 (coe
                                                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                    (coe v14)
                                                    (coe
                                                       MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28
                                                       (coe
                                                          MAlonzo.Code.Ar.du_splitP_172 (coe v5)
                                                          (coe v15))))
                                                 (MAlonzo.Code.Agda.Builtin.Sigma.d_snd_30
                                                    (coe
                                                       MAlonzo.Code.Ar.du_splitP_172 (coe v5)
                                                       (coe v15))))) in
                               coe
                                 (case coe v12 of
                                    MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                      -> case coe v13 of
                                           MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v15
                                             -> case coe v15 of
                                                  MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                                                    -> case coe v17 of
                                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v18 v19
                                                           -> case coe v19 of
                                                                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v20 v21
                                                                  -> let v22
                                                                           = MAlonzo.Code.Lang.d_stren'45''8707'_1122
                                                                               (coe
                                                                                  MAlonzo.Code.Lang.C__'9657'__40
                                                                                  (coe v1)
                                                                                  (coe
                                                                                     MAlonzo.Code.Lang.C_ix_32
                                                                                     (coe v5)))
                                                                               (coe
                                                                                  MAlonzo.Code.Lang.C_ar_34
                                                                                  (coe v16))
                                                                               (coe
                                                                                  MAlonzo.Code.Lang.C_ix_32
                                                                                  (coe v5))
                                                                               (coe v18)
                                                                               (coe
                                                                                  MAlonzo.Code.Lang.C_here_60) in
                                                                     coe
                                                                       (case coe v22 of
                                                                          MAlonzo.Code.Agda.Builtin.Maybe.C_just_16 v23
                                                                            -> case coe v23 of
                                                                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v24 v25
                                                                                   -> coe
                                                                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                        (coe
                                                                                           MAlonzo.Code.Lang.C_let'8242'_246
                                                                                           v16 v24
                                                                                           (coe
                                                                                              MAlonzo.Code.Lang.C_imap_226
                                                                                              v5 v6
                                                                                              (MAlonzo.Code.Lang.d_sub_554
                                                                                                 (coe
                                                                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                                                                    (coe
                                                                                                       MAlonzo.Code.Lang.C__'9657'__40
                                                                                                       (coe
                                                                                                          v1)
                                                                                                       (coe
                                                                                                          MAlonzo.Code.Lang.C_ix_32
                                                                                                          (coe
                                                                                                             v5)))
                                                                                                    (coe
                                                                                                       MAlonzo.Code.Lang.C_ar_34
                                                                                                       (coe
                                                                                                          v16)))
                                                                                                 (coe
                                                                                                    MAlonzo.Code.Lang.C_ar_34
                                                                                                    (coe
                                                                                                       v6))
                                                                                                 (coe
                                                                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                                                                    (coe
                                                                                                       MAlonzo.Code.Lang.C__'9657'__40
                                                                                                       (coe
                                                                                                          v1)
                                                                                                       (coe
                                                                                                          MAlonzo.Code.Lang.C_ar_34
                                                                                                          (coe
                                                                                                             v16)))
                                                                                                    (coe
                                                                                                       MAlonzo.Code.Lang.C_ix_32
                                                                                                       (coe
                                                                                                          v5)))
                                                                                                 (coe
                                                                                                    v20)
                                                                                                 (coe
                                                                                                    MAlonzo.Code.Lang.d_sub'45'swap_840
                                                                                                    (coe
                                                                                                       v1)
                                                                                                    (coe
                                                                                                       MAlonzo.Code.Lang.C_ar_34
                                                                                                       (coe
                                                                                                          v16))
                                                                                                    (coe
                                                                                                       MAlonzo.Code.Lang.C_ix_32
                                                                                                       (coe
                                                                                                          v5))))))
                                                                                        erased
                                                                                 _ -> MAlonzo.RTE.mazUnreachableError
                                                                          MAlonzo.Code.Agda.Builtin.Maybe.C_nothing_18
                                                                            -> coe
                                                                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                 (coe
                                                                                    MAlonzo.Code.Lang.C_imap_226
                                                                                    v5 v6
                                                                                    (coe
                                                                                       MAlonzo.Code.Lang.C_let'8242'_246
                                                                                       v16 v18 v20))
                                                                                 (coe
                                                                                    (\ v23 v24 ->
                                                                                       coe
                                                                                         v10
                                                                                         (coe
                                                                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                            (coe
                                                                                               v23)
                                                                                            (coe
                                                                                               MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28
                                                                                               (coe
                                                                                                  MAlonzo.Code.Ar.du_splitP_172
                                                                                                  (coe
                                                                                                     v5)
                                                                                                  (coe
                                                                                                     v24))))
                                                                                         (MAlonzo.Code.Agda.Builtin.Sigma.d_snd_30
                                                                                            (coe
                                                                                               MAlonzo.Code.Ar.du_splitP_172
                                                                                               (coe
                                                                                                  v5)
                                                                                               (coe
                                                                                                  v24)))))
                                                                          _ -> MAlonzo.RTE.mazUnreachableError)
                                                                _ -> MAlonzo.RTE.mazUnreachableError
                                                         _ -> MAlonzo.RTE.mazUnreachableError
                                                  _ -> MAlonzo.RTE.mazUnreachableError
                                           _ -> coe v14
                                    _ -> coe v14)
                          _ -> MAlonzo.RTE.mazUnreachableError)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_sel_228 v5 v7 v8
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v9
               -> let v10
                        = coe
                            du_opt_214 (coe v0) (coe v1)
                            (coe
                               MAlonzo.Code.Lang.C_ar_34
                               (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v5 v9))
                            (coe v7) in
                  coe
                    (let v11
                           = coe
                               du_opt_214 (coe v0) (coe v1)
                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) (coe v8) in
                     coe
                       (case coe v10 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                            -> case coe v11 of
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                   -> let v16 = coe MAlonzo.Code.LangEq.du_isZero_144 (coe v12) in
                                      coe
                                        (case coe v16 of
                                           MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v17 v18
                                             -> if coe v17
                                                  then coe
                                                         seq (coe v18)
                                                         (coe
                                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                            (coe MAlonzo.Code.Lang.C_zero_218)
                                                            (coe
                                                               (\ v19 v20 ->
                                                                  coe
                                                                    v13 v19
                                                                    (coe
                                                                       MAlonzo.Code.Ar.du__'43''43'__56
                                                                       (coe v5)
                                                                       (coe
                                                                          MAlonzo.Code.Eval.d_eval_114
                                                                          (coe v0) (coe v1)
                                                                          (coe
                                                                             MAlonzo.Code.Lang.C_ix_32
                                                                             (coe v5))
                                                                          (coe v8) (coe v19))
                                                                       (coe v20)))))
                                                  else coe
                                                         seq (coe v18)
                                                         (let v19
                                                                = coe
                                                                    MAlonzo.Code.LangEq.du_isOne_216
                                                                    (coe v12) in
                                                          coe
                                                            (case coe v19 of
                                                               MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v20 v21
                                                                 -> if coe v20
                                                                      then coe
                                                                             seq (coe v21)
                                                                             (coe
                                                                                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                (coe
                                                                                   MAlonzo.Code.Lang.C_one_220)
                                                                                (coe
                                                                                   (\ v22 v23 ->
                                                                                      coe
                                                                                        v13 v22
                                                                                        (coe
                                                                                           MAlonzo.Code.Ar.du__'43''43'__56
                                                                                           (coe v5)
                                                                                           (coe
                                                                                              MAlonzo.Code.Eval.d_eval_114
                                                                                              (coe
                                                                                                 v0)
                                                                                              (coe
                                                                                                 v1)
                                                                                              (coe
                                                                                                 MAlonzo.Code.Lang.C_ix_32
                                                                                                 (coe
                                                                                                    v5))
                                                                                              (coe
                                                                                                 v8)
                                                                                              (coe
                                                                                                 v22))
                                                                                           (coe
                                                                                              v23)))))
                                                                      else coe
                                                                             seq (coe v21)
                                                                             (let v22
                                                                                    = coe
                                                                                        MAlonzo.Code.LangEq.du_isImap_296
                                                                                        (coe v12) in
                                                                              coe
                                                                                (case coe v22 of
                                                                                   MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v23 v24
                                                                                     -> if coe v23
                                                                                          then case coe
                                                                                                      v24 of
                                                                                                 MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v25
                                                                                                   -> case coe
                                                                                                             v25 of
                                                                                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v26 v27
                                                                                                          -> case coe
                                                                                                                    v27 of
                                                                                                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v28 v29
                                                                                                                 -> case coe
                                                                                                                           v29 of
                                                                                                                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v30 v31
                                                                                                                        -> case coe
                                                                                                                                  v31 of
                                                                                                                             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v32 v33
                                                                                                                               -> let v34
                                                                                                                                        = MAlonzo.Code.Ar.d__'8799''738'__70
                                                                                                                                            (coe
                                                                                                                                               v5)
                                                                                                                                            (coe
                                                                                                                                               v26) in
                                                                                                                                  coe
                                                                                                                                    (case coe
                                                                                                                                            v34 of
                                                                                                                                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v35 v36
                                                                                                                                         -> if coe
                                                                                                                                                 v35
                                                                                                                                              then coe
                                                                                                                                                     seq
                                                                                                                                                     (coe
                                                                                                                                                        v36)
                                                                                                                                                     (coe
                                                                                                                                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                                        (coe
                                                                                                                                                           MAlonzo.Code.Lang.d_sub_554
                                                                                                                                                           (coe
                                                                                                                                                              MAlonzo.Code.Lang.C__'9657'__40
                                                                                                                                                              (coe
                                                                                                                                                                 v1)
                                                                                                                                                              (coe
                                                                                                                                                                 MAlonzo.Code.Lang.C_ix_32
                                                                                                                                                                 (coe
                                                                                                                                                                    v26)))
                                                                                                                                                           (coe
                                                                                                                                                              MAlonzo.Code.Lang.C_ar_34
                                                                                                                                                              (coe
                                                                                                                                                                 v28))
                                                                                                                                                           (coe
                                                                                                                                                              v1)
                                                                                                                                                           (coe
                                                                                                                                                              v32)
                                                                                                                                                           (coe
                                                                                                                                                              MAlonzo.Code.Lang.C__'9657'__516
                                                                                                                                                              (MAlonzo.Code.Lang.d_sub'45'id_548
                                                                                                                                                                 (coe
                                                                                                                                                                    v1))
                                                                                                                                                              v14))
                                                                                                                                                        erased)
                                                                                                                                              else coe
                                                                                                                                                     seq
                                                                                                                                                     (coe
                                                                                                                                                        v36)
                                                                                                                                                     (coe
                                                                                                                                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                                        (coe
                                                                                                                                                           MAlonzo.Code.Lang.C_sel_228
                                                                                                                                                           v5
                                                                                                                                                           v12
                                                                                                                                                           v14)
                                                                                                                                                        erased)
                                                                                                                                       _ -> MAlonzo.RTE.mazUnreachableError)
                                                                                                                             _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                      _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                               _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                        _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                 _ -> MAlonzo.RTE.mazUnreachableError
                                                                                          else coe
                                                                                                 seq
                                                                                                 (coe
                                                                                                    v24)
                                                                                                 (let v25
                                                                                                        = coe
                                                                                                            MAlonzo.Code.LangEq.du_isZeroBut_484
                                                                                                            (coe
                                                                                                               v12) in
                                                                                                  coe
                                                                                                    (case coe
                                                                                                            v25 of
                                                                                                       MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v26 v27
                                                                                                         -> if coe
                                                                                                                 v26
                                                                                                              then case coe
                                                                                                                          v27 of
                                                                                                                     MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v28
                                                                                                                       -> case coe
                                                                                                                                 v28 of
                                                                                                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v29 v30
                                                                                                                              -> case coe
                                                                                                                                        v30 of
                                                                                                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v31 v32
                                                                                                                                     -> case coe
                                                                                                                                               v32 of
                                                                                                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v33 v34
                                                                                                                                            -> case coe
                                                                                                                                                      v34 of
                                                                                                                                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v35 v36
                                                                                                                                                   -> coe
                                                                                                                                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                                        (coe
                                                                                                                                                           MAlonzo.Code.Lang.C_zero'45'but_236
                                                                                                                                                           v29
                                                                                                                                                           v31
                                                                                                                                                           v33
                                                                                                                                                           (coe
                                                                                                                                                              MAlonzo.Code.Lang.C_sel_228
                                                                                                                                                              v5
                                                                                                                                                              v35
                                                                                                                                                              v14))
                                                                                                                                                        erased
                                                                                                                                                 _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                          _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                            _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                     _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                              else coe
                                                                                                                     seq
                                                                                                                     (coe
                                                                                                                        v27)
                                                                                                                     (let v28
                                                                                                                            = coe
                                                                                                                                MAlonzo.Code.LangEq.du_isLet_1704
                                                                                                                                (coe
                                                                                                                                   v12) in
                                                                                                                      coe
                                                                                                                        (case coe
                                                                                                                                v28 of
                                                                                                                           MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v29 v30
                                                                                                                             -> if coe
                                                                                                                                     v29
                                                                                                                                  then case coe
                                                                                                                                              v30 of
                                                                                                                                         MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v31
                                                                                                                                           -> case coe
                                                                                                                                                     v31 of
                                                                                                                                                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v32 v33
                                                                                                                                                  -> case coe
                                                                                                                                                            v33 of
                                                                                                                                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v34 v35
                                                                                                                                                         -> case coe
                                                                                                                                                                   v35 of
                                                                                                                                                              MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v36 v37
                                                                                                                                                                -> coe
                                                                                                                                                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                                                     (coe
                                                                                                                                                                        MAlonzo.Code.Lang.C_let'8242'_246
                                                                                                                                                                        v32
                                                                                                                                                                        v34
                                                                                                                                                                        (coe
                                                                                                                                                                           MAlonzo.Code.Lang.C_sel_228
                                                                                                                                                                           v5
                                                                                                                                                                           v36
                                                                                                                                                                           (coe
                                                                                                                                                                              MAlonzo.Code.Lang.d__'8593'_500
                                                                                                                                                                              v1
                                                                                                                                                                              (coe
                                                                                                                                                                                 MAlonzo.Code.Lang.C_ix_32
                                                                                                                                                                                 (coe
                                                                                                                                                                                    v5))
                                                                                                                                                                              (coe
                                                                                                                                                                                 MAlonzo.Code.Lang.C_ar_34
                                                                                                                                                                                 (coe
                                                                                                                                                                                    v32))
                                                                                                                                                                              v14)))
                                                                                                                                                                     erased
                                                                                                                                                              _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                       _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                         _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                  else coe
                                                                                                                                         seq
                                                                                                                                         (coe
                                                                                                                                            v30)
                                                                                                                                         (coe
                                                                                                                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                            (coe
                                                                                                                                               MAlonzo.Code.Lang.C_sel_228
                                                                                                                                               v5
                                                                                                                                               v12
                                                                                                                                               v14)
                                                                                                                                            erased)
                                                                                                                           _ -> MAlonzo.RTE.mazUnreachableError))
                                                                                                       _ -> MAlonzo.RTE.mazUnreachableError))
                                                                                   _ -> MAlonzo.RTE.mazUnreachableError))
                                                               _ -> MAlonzo.RTE.mazUnreachableError))
                                           _ -> MAlonzo.RTE.mazUnreachableError)
                                 _ -> MAlonzo.RTE.mazUnreachableError
                          _ -> MAlonzo.RTE.mazUnreachableError))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_imapb_230 v4 v5 v8 v9
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v10
               -> let v11
                        = coe
                            du_opt_214 (coe v0)
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)))
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)) (coe v9) in
                  coe
                    (case coe v11 of
                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                         -> coe
                              MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                              (coe MAlonzo.Code.Lang.C_imapb_230 v4 v5 v8 v12)
                              (coe
                                 (\ v14 v15 ->
                                    coe
                                      v13
                                      (coe
                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v14)
                                         (coe
                                            MAlonzo.Code.Ar.d_ix'45'div_1238 (coe v10) (coe v4)
                                            (coe v5) (coe v15) (coe v8)))
                                      (MAlonzo.Code.Ar.d_ix'45'mod_1248
                                         (coe v10) (coe v4) (coe v5) (coe v15) (coe v8))))
                       _ -> MAlonzo.RTE.mazUnreachableError)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_selb_232 v4 v6 v8 v9 v10
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v11
               -> let v12
                        = coe
                            du_opt_214 (coe v0) (coe v1)
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe v9) in
                  coe
                    (let v13
                           = coe
                               du_opt_214 (coe v0) (coe v1)
                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v4)) (coe v10) in
                     coe
                       (case coe v12 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                            -> case coe v13 of
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                                   -> coe
                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                        (coe MAlonzo.Code.Lang.C_selb_232 v4 v6 v8 v14 v16) erased
                                 _ -> MAlonzo.RTE.mazUnreachableError
                          _ -> MAlonzo.RTE.mazUnreachableError))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sum_234 v5 v7
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v8
               -> let v9
                        = coe
                            du_opt_214 (coe v0)
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                            (coe v2) (coe v7) in
                  coe
                    (case coe v9 of
                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v10 v11
                         -> let v12
                                  = coe
                                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                      (coe MAlonzo.Code.Lang.C_sum_234 v5 v10) erased in
                            coe
                              (case coe v10 of
                                 MAlonzo.Code.Lang.C_zero_218
                                   -> coe
                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                        (coe MAlonzo.Code.Lang.C_zero_218) erased
                                 MAlonzo.Code.Lang.C_imaps_222 v15
                                   -> coe
                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                        (coe
                                           MAlonzo.Code.Lang.C_imaps_222
                                           (coe
                                              MAlonzo.Code.Lang.C_sum_234 v5
                                              (MAlonzo.Code.Lang.d_sub_554
                                                 (coe
                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                    (coe
                                                       MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v8)))
                                                 (coe
                                                    MAlonzo.Code.Lang.C_ar_34
                                                    (coe MAlonzo.Code.Lang.d_unit_212))
                                                 (coe
                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                    (coe
                                                       MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v8)))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                                 (coe v15)
                                                 (coe
                                                    MAlonzo.Code.Lang.d_sub'45'swap_840 (coe v1)
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v8))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v5))))))
                                        erased
                                 MAlonzo.Code.Lang.C_imap_226 v14 v15 v16
                                   -> coe
                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                        (coe
                                           MAlonzo.Code.Lang.C_imap_226 v14 v15
                                           (coe
                                              MAlonzo.Code.Lang.C_sum_234 v5
                                              (MAlonzo.Code.Lang.d_sub_554
                                                 (coe
                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                    (coe
                                                       MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v14)))
                                                 (coe MAlonzo.Code.Lang.C_ar_34 (coe v15))
                                                 (coe
                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                    (coe
                                                       MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v14)))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                                 (coe v16)
                                                 (coe
                                                    MAlonzo.Code.Lang.d_sub'45'swap_840 (coe v1)
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v14))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v5))))))
                                        erased
                                 MAlonzo.Code.Lang.C_imapb_230 v13 v14 v17 v18
                                   -> coe
                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                        (coe
                                           MAlonzo.Code.Lang.C_imapb_230 v13 v14 v17
                                           (coe
                                              MAlonzo.Code.Lang.C_sum_234 v5
                                              (MAlonzo.Code.Lang.d_sub_554
                                                 (coe
                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                    (coe
                                                       MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v13)))
                                                 (coe MAlonzo.Code.Lang.C_ar_34 (coe v14))
                                                 (coe
                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                    (coe
                                                       MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v13)))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                                 (coe v18)
                                                 (coe
                                                    MAlonzo.Code.Lang.d_sub'45'swap_840 (coe v1)
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v13))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v5))))))
                                        erased
                                 MAlonzo.Code.Lang.C_zero'45'but_236 v14 v16 v17 v18
                                   -> case coe v16 of
                                        MAlonzo.Code.Lang.C_var_216 v21
                                          -> case coe v17 of
                                               MAlonzo.Code.Lang.C_var_216 v24
                                                 -> let v25
                                                          = coe
                                                              MAlonzo.Code.Lang.du_eq'63'_142
                                                              (coe
                                                                 MAlonzo.Code.Lang.C__'9657'__40
                                                                 (coe v1)
                                                                 (coe
                                                                    MAlonzo.Code.Lang.C_ix_32
                                                                    (coe v5)))
                                                              (coe MAlonzo.Code.Lang.C_here_60)
                                                              (coe v21) in
                                                    coe
                                                      (let v26
                                                             = coe
                                                                 MAlonzo.Code.Lang.du_eq'63'_142
                                                                 (coe
                                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                                    (coe v1)
                                                                    (coe
                                                                       MAlonzo.Code.Lang.C_ix_32
                                                                       (coe v5)))
                                                                 (coe MAlonzo.Code.Lang.C_here_60)
                                                                 (coe v24) in
                                                       coe
                                                         (case coe v25 of
                                                            MAlonzo.Code.Lang.C_veq_130
                                                              -> case coe v26 of
                                                                   MAlonzo.Code.Lang.C_veq_130
                                                                     -> coe
                                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                          (coe
                                                                             MAlonzo.Code.Lang.C_sum_234
                                                                             v5 v18)
                                                                          erased
                                                                   MAlonzo.Code.Lang.C_neq_136 v34
                                                                     -> coe
                                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                          (coe
                                                                             MAlonzo.Code.Lang.d_sub_554
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C__'9657'__40
                                                                                (coe v1)
                                                                                (coe
                                                                                   MAlonzo.Code.Lang.C_ix_32
                                                                                   (coe v5)))
                                                                             (coe v2) (coe v1)
                                                                             (coe v18)
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C__'9657'__516
                                                                                (MAlonzo.Code.Lang.d_sub'45'id_548
                                                                                   (coe v1))
                                                                                (coe
                                                                                   MAlonzo.Code.Lang.C_var_216
                                                                                   v34)))
                                                                          erased
                                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                                            MAlonzo.Code.Lang.C_neq_136 v31
                                                              -> case coe v26 of
                                                                   MAlonzo.Code.Lang.C_veq_130
                                                                     -> coe
                                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                          (coe
                                                                             MAlonzo.Code.Lang.d_sub_554
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C__'9657'__40
                                                                                (coe v1)
                                                                                (coe
                                                                                   MAlonzo.Code.Lang.C_ix_32
                                                                                   (coe v5)))
                                                                             (coe v2) (coe v1)
                                                                             (coe v18)
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C__'9657'__516
                                                                                (MAlonzo.Code.Lang.d_sub'45'id_548
                                                                                   (coe v1))
                                                                                (coe
                                                                                   MAlonzo.Code.Lang.C_var_216
                                                                                   v31)))
                                                                          erased
                                                                   MAlonzo.Code.Lang.C_neq_136 v36
                                                                     -> coe
                                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                          (coe
                                                                             MAlonzo.Code.Lang.C_zero'45'but_236
                                                                             v14
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C_var_216
                                                                                v31)
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C_var_216
                                                                                v36)
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C_sum_234
                                                                                v5 v18))
                                                                          erased
                                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                                            _ -> MAlonzo.RTE.mazUnreachableError))
                                               _ -> coe v12
                                        _ -> coe v12
                                 MAlonzo.Code.Lang.C_let'8242'_246 v14 v16 v17
                                   -> coe
                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                        (coe
                                           MAlonzo.Code.Lang.C_let'8242'_246
                                           (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v5 v14)
                                           (coe MAlonzo.Code.Lang.C_imap_226 v5 v14 v16)
                                           (coe
                                              MAlonzo.Code.Lang.C_sum_234 v5
                                              (MAlonzo.Code.Lang.d_sub_554
                                                 (coe
                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                    (coe
                                                       MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                                    (coe MAlonzo.Code.Lang.C_ar_34 (coe v14)))
                                                 (coe v2)
                                                 (coe
                                                    MAlonzo.Code.Lang.C__'9657'__40
                                                    (coe
                                                       MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                       (coe
                                                          MAlonzo.Code.Lang.C_ar_34
                                                          (coe
                                                             MAlonzo.Code.Ar.d__'8855'__54 () erased
                                                             v5 v14)))
                                                    (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                                                 (coe v17)
                                                 (coe
                                                    MAlonzo.Code.Lang.C__'9657'__516
                                                    (MAlonzo.Code.Lang.d_skeep_532
                                                       (coe
                                                          MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                                          (coe
                                                             MAlonzo.Code.Lang.C_ar_34
                                                             (coe
                                                                MAlonzo.Code.Ar.d__'8855'__54 ()
                                                                erased v5 v14)))
                                                       (coe v1)
                                                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v5))
                                                       (coe
                                                          MAlonzo.Code.Lang.d_sdrop_528 (coe v1)
                                                          (coe v1)
                                                          (coe
                                                             MAlonzo.Code.Lang.C_ar_34
                                                             (coe
                                                                MAlonzo.Code.Ar.d__'8855'__54 ()
                                                                erased v5 v14))
                                                          (coe
                                                             MAlonzo.Code.Lang.d_sub'45'id_548
                                                             (coe v1))))
                                                    (coe
                                                       MAlonzo.Code.Lang.C_sel_228 v5
                                                       (coe
                                                          MAlonzo.Code.Lang.C_var_216
                                                          (coe
                                                             MAlonzo.Code.Lang.C_there_62
                                                             (coe MAlonzo.Code.Lang.C_here_60)))
                                                       (coe
                                                          MAlonzo.Code.Lang.C_var_216
                                                          (coe MAlonzo.Code.Lang.C_here_60)))))))
                                        erased
                                 _ -> coe v12)
                       _ -> MAlonzo.RTE.mazUnreachableError)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_zero'45'but_236 v5 v7 v8 v9
        -> let v10 = coe du_opt_214 (coe v0) (coe v1) (coe v2) (coe v9) in
           coe
             (case coe v10 of
                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v11 v12
                  -> coe
                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                       (coe MAlonzo.Code.Lang.C_zero'45'but_236 v5 v7 v8 v11) erased
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_slide_238 v5 v6 v7 v9 v10 v11 v12
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v13
               -> let v14
                        = coe
                            du_opt_214 (coe v0) (coe v1)
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v7)) (coe v11) in
                  coe
                    (case coe v14 of
                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v15 v16
                         -> coe
                              MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                              (coe MAlonzo.Code.Lang.C_slide_238 v5 v6 v7 v9 v10 v15 v12)
                              (coe
                                 (\ v17 v18 ->
                                    coe
                                      v16 v17
                                      (MAlonzo.Code.Ar.d__'8853''8242'__1052
                                         (coe v5) (coe v13) (coe v6) (coe v7)
                                         (coe
                                            MAlonzo.Code.Eval.d_eval_114 (coe v0) (coe v1)
                                            (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) (coe v9)
                                            (coe v17))
                                         (coe v18) (coe v12) (coe v10))))
                       _ -> MAlonzo.RTE.mazUnreachableError)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_backslide_240 v5 v6 v7 v9 v10 v11 v12
        -> let v13
                 = coe
                     du_opt_214 (coe v0) (coe v1)
                     (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe v10) in
           coe
             (case coe v13 of
                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                  -> coe
                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                       (coe MAlonzo.Code.Lang.C_backslide_240 v5 v6 v7 v9 v14 v11 v12)
                       erased
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_bin_242 v6 v7 v8
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v9
               -> case coe v6 of
                    MAlonzo.Code.Lang.C_plus_190
                      -> let v10 = coe du_opt_214 (coe v0) (coe v1) (coe v2) (coe v7) in
                         coe
                           (let v11 = coe du_opt_214 (coe v0) (coe v1) (coe v2) (coe v8) in
                            coe
                              (case coe v10 of
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                   -> case coe v11 of
                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                          -> let v16
                                                   = coe
                                                       MAlonzo.Code.LangEq.du_isZero_144
                                                       (coe v12) in
                                             coe
                                               (case coe v16 of
                                                  MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v17 v18
                                                    -> if coe v17
                                                         then coe
                                                                seq (coe v18)
                                                                (coe
                                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                   (coe v14) erased)
                                                         else coe
                                                                seq (coe v18)
                                                                (let v19
                                                                       = coe
                                                                           MAlonzo.Code.LangEq.du_isZero_144
                                                                           (coe v14) in
                                                                 coe
                                                                   (case coe v19 of
                                                                      MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v20 v21
                                                                        -> if coe v20
                                                                             then coe
                                                                                    seq (coe v21)
                                                                                    (coe
                                                                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                       (coe v12)
                                                                                       erased)
                                                                             else coe
                                                                                    seq (coe v21)
                                                                                    (let v22
                                                                                           = coe
                                                                                               MAlonzo.Code.LangEq.du_isImaps_404
                                                                                               (coe
                                                                                                  v12) in
                                                                                     coe
                                                                                       (case coe
                                                                                               v22 of
                                                                                          MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v23 v24
                                                                                            -> if coe
                                                                                                    v23
                                                                                                 then case coe
                                                                                                             v24 of
                                                                                                        MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v25
                                                                                                          -> case coe
                                                                                                                    v25 of
                                                                                                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v26 v27
                                                                                                                 -> coe
                                                                                                                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                      (coe
                                                                                                                         MAlonzo.Code.Lang.C_imaps_222
                                                                                                                         (coe
                                                                                                                            MAlonzo.Code.Lang.C_bin_242
                                                                                                                            v6
                                                                                                                            v26
                                                                                                                            (coe
                                                                                                                               MAlonzo.Code.Lang.C_sels_224
                                                                                                                               v9
                                                                                                                               (coe
                                                                                                                                  MAlonzo.Code.Lang.d__'8593'_500
                                                                                                                                  v1
                                                                                                                                  v2
                                                                                                                                  (coe
                                                                                                                                     MAlonzo.Code.Lang.C_ix_32
                                                                                                                                     (coe
                                                                                                                                        v9))
                                                                                                                                  v14)
                                                                                                                               (coe
                                                                                                                                  MAlonzo.Code.Lang.C_var_216
                                                                                                                                  (coe
                                                                                                                                     MAlonzo.Code.Lang.C_here_60)))))
                                                                                                                      erased
                                                                                                               _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                        _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                 else coe
                                                                                                        seq
                                                                                                        (coe
                                                                                                           v24)
                                                                                                        (let v25
                                                                                                               = coe
                                                                                                                   MAlonzo.Code.LangEq.du_isImaps_404
                                                                                                                   (coe
                                                                                                                      v14) in
                                                                                                         coe
                                                                                                           (case coe
                                                                                                                   v25 of
                                                                                                              MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v26 v27
                                                                                                                -> if coe
                                                                                                                        v26
                                                                                                                     then case coe
                                                                                                                                 v27 of
                                                                                                                            MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v28
                                                                                                                              -> case coe
                                                                                                                                        v28 of
                                                                                                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v29 v30
                                                                                                                                     -> coe
                                                                                                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                          (coe
                                                                                                                                             MAlonzo.Code.Lang.C_imaps_222
                                                                                                                                             (coe
                                                                                                                                                MAlonzo.Code.Lang.C_bin_242
                                                                                                                                                v6
                                                                                                                                                (coe
                                                                                                                                                   MAlonzo.Code.Lang.C_sels_224
                                                                                                                                                   v9
                                                                                                                                                   (coe
                                                                                                                                                      MAlonzo.Code.Lang.d__'8593'_500
                                                                                                                                                      v1
                                                                                                                                                      v2
                                                                                                                                                      (coe
                                                                                                                                                         MAlonzo.Code.Lang.C_ix_32
                                                                                                                                                         (coe
                                                                                                                                                            v9))
                                                                                                                                                      v12)
                                                                                                                                                   (coe
                                                                                                                                                      MAlonzo.Code.Lang.C_var_216
                                                                                                                                                      (coe
                                                                                                                                                         MAlonzo.Code.Lang.C_here_60)))
                                                                                                                                                v29))
                                                                                                                                          erased
                                                                                                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                            _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                     else (case coe
                                                                                                                                  v27 of
                                                                                                                             MAlonzo.Code.Relation.Nullary.Reflects.C_of'8319'_26
                                                                                                                               -> let v29
                                                                                                                                        = coe
                                                                                                                                            MAlonzo.Code.LangEq.du_isImap_296
                                                                                                                                            (coe
                                                                                                                                               v12) in
                                                                                                                                  coe
                                                                                                                                    (let v30
                                                                                                                                           = coe
                                                                                                                                               MAlonzo.Code.LangEq.du_isImap_296
                                                                                                                                               (coe
                                                                                                                                                  v14) in
                                                                                                                                     coe
                                                                                                                                       (case coe
                                                                                                                                               v29 of
                                                                                                                                          MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v31 v32
                                                                                                                                            -> let v33
                                                                                                                                                     = let v33
                                                                                                                                                             = let v33
                                                                                                                                                                     = coe
                                                                                                                                                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                                                         (coe
                                                                                                                                                                            MAlonzo.Code.Lang.C_bin_242
                                                                                                                                                                            v6
                                                                                                                                                                            v12
                                                                                                                                                                            v14)
                                                                                                                                                                         erased in
                                                                                                                                                               coe
                                                                                                                                                                 (case coe
                                                                                                                                                                         v14 of
                                                                                                                                                                    MAlonzo.Code.Lang.C_zero'45'but_236 v35 v37 v38 v39
                                                                                                                                                                      -> case coe
                                                                                                                                                                                v37 of
                                                                                                                                                                           MAlonzo.Code.Lang.C_var_216 v42
                                                                                                                                                                             -> case coe
                                                                                                                                                                                       v38 of
                                                                                                                                                                                  MAlonzo.Code.Lang.C_var_216 v45
                                                                                                                                                                                    -> case coe
                                                                                                                                                                                              v12 of
                                                                                                                                                                                         MAlonzo.Code.Lang.C_zero'45'but_236 v47 v49 v50 v51
                                                                                                                                                                                           -> case coe
                                                                                                                                                                                                     v49 of
                                                                                                                                                                                                MAlonzo.Code.Lang.C_var_216 v54
                                                                                                                                                                                                  -> case coe
                                                                                                                                                                                                            v50 of
                                                                                                                                                                                                       MAlonzo.Code.Lang.C_var_216 v57
                                                                                                                                                                                                         -> let v58
                                                                                                                                                                                                                  = coe
                                                                                                                                                                                                                      MAlonzo.Code.Lang.du_eq'63'_142
                                                                                                                                                                                                                      (coe
                                                                                                                                                                                                                         v1)
                                                                                                                                                                                                                      (coe
                                                                                                                                                                                                                         v54)
                                                                                                                                                                                                                      (coe
                                                                                                                                                                                                                         v42) in
                                                                                                                                                                                                            coe
                                                                                                                                                                                                              (let v59
                                                                                                                                                                                                                     = coe
                                                                                                                                                                                                                         MAlonzo.Code.Lang.du_eq'63'_142
                                                                                                                                                                                                                         (coe
                                                                                                                                                                                                                            v1)
                                                                                                                                                                                                                         (coe
                                                                                                                                                                                                                            v57)
                                                                                                                                                                                                                         (coe
                                                                                                                                                                                                                            v45) in
                                                                                                                                                                                                               coe
                                                                                                                                                                                                                 (let v60
                                                                                                                                                                                                                        = coe
                                                                                                                                                                                                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                                                                                                            (coe
                                                                                                                                                                                                                               MAlonzo.Code.Lang.C_bin_242
                                                                                                                                                                                                                               v6
                                                                                                                                                                                                                               (coe
                                                                                                                                                                                                                                  MAlonzo.Code.Lang.C_zero'45'but_236
                                                                                                                                                                                                                                  v47
                                                                                                                                                                                                                                  (coe
                                                                                                                                                                                                                                     MAlonzo.Code.Lang.C_var_216
                                                                                                                                                                                                                                     v54)
                                                                                                                                                                                                                                  (coe
                                                                                                                                                                                                                                     MAlonzo.Code.Lang.C_var_216
                                                                                                                                                                                                                                     v57)
                                                                                                                                                                                                                                  v51)
                                                                                                                                                                                                                               (coe
                                                                                                                                                                                                                                  MAlonzo.Code.Lang.C_zero'45'but_236
                                                                                                                                                                                                                                  v35
                                                                                                                                                                                                                                  (coe
                                                                                                                                                                                                                                     MAlonzo.Code.Lang.C_var_216
                                                                                                                                                                                                                                     v42)
                                                                                                                                                                                                                                  (coe
                                                                                                                                                                                                                                     MAlonzo.Code.Lang.C_var_216
                                                                                                                                                                                                                                     v45)
                                                                                                                                                                                                                                  v39))
                                                                                                                                                                                                                            erased in
                                                                                                                                                                                                                  coe
                                                                                                                                                                                                                    (case coe
                                                                                                                                                                                                                            v58 of
                                                                                                                                                                                                                       MAlonzo.Code.Lang.C_veq_130
                                                                                                                                                                                                                         -> case coe
                                                                                                                                                                                                                                   v59 of
                                                                                                                                                                                                                              MAlonzo.Code.Lang.C_veq_130
                                                                                                                                                                                                                                -> coe
                                                                                                                                                                                                                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                                                                                                                     (coe
                                                                                                                                                                                                                                        MAlonzo.Code.Lang.C_zero'45'but_236
                                                                                                                                                                                                                                        v35
                                                                                                                                                                                                                                        (coe
                                                                                                                                                                                                                                           MAlonzo.Code.Lang.C_var_216
                                                                                                                                                                                                                                           v42)
                                                                                                                                                                                                                                        (coe
                                                                                                                                                                                                                                           MAlonzo.Code.Lang.C_var_216
                                                                                                                                                                                                                                           v45)
                                                                                                                                                                                                                                        (coe
                                                                                                                                                                                                                                           MAlonzo.Code.Lang.C_bin_242
                                                                                                                                                                                                                                           v6
                                                                                                                                                                                                                                           v51
                                                                                                                                                                                                                                           v39))
                                                                                                                                                                                                                                     erased
                                                                                                                                                                                                                              _ -> coe
                                                                                                                                                                                                                                     v60
                                                                                                                                                                                                                       _ -> coe
                                                                                                                                                                                                                              v60)))
                                                                                                                                                                                                       _ -> coe
                                                                                                                                                                                                              v33
                                                                                                                                                                                                _ -> coe
                                                                                                                                                                                                       v33
                                                                                                                                                                                         _ -> coe
                                                                                                                                                                                                v33
                                                                                                                                                                                  _ -> coe
                                                                                                                                                                                         v33
                                                                                                                                                                           _ -> coe
                                                                                                                                                                                  v33
                                                                                                                                                                    _ -> coe
                                                                                                                                                                           v33) in
                                                                                                                                                       coe
                                                                                                                                                         (case coe
                                                                                                                                                                 v30 of
                                                                                                                                                            MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v34 v35
                                                                                                                                                              -> case coe
                                                                                                                                                                        v34 of
                                                                                                                                                                   MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                                                                                                                                                     -> case coe
                                                                                                                                                                               v35 of
                                                                                                                                                                          MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v36
                                                                                                                                                                            -> case coe
                                                                                                                                                                                      v36 of
                                                                                                                                                                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v37 v38
                                                                                                                                                                                   -> case coe
                                                                                                                                                                                             v38 of
                                                                                                                                                                                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v39 v40
                                                                                                                                                                                          -> case coe
                                                                                                                                                                                                    v40 of
                                                                                                                                                                                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v41 v42
                                                                                                                                                                                                 -> case coe
                                                                                                                                                                                                           v42 of
                                                                                                                                                                                                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v43 v44
                                                                                                                                                                                                        -> coe
                                                                                                                                                                                                             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                                                                                             (coe
                                                                                                                                                                                                                MAlonzo.Code.Lang.C_imap_226
                                                                                                                                                                                                                v37
                                                                                                                                                                                                                v39
                                                                                                                                                                                                                (coe
                                                                                                                                                                                                                   MAlonzo.Code.Lang.C_bin_242
                                                                                                                                                                                                                   v6
                                                                                                                                                                                                                   (coe
                                                                                                                                                                                                                      MAlonzo.Code.Lang.C_sel_228
                                                                                                                                                                                                                      v37
                                                                                                                                                                                                                      (coe
                                                                                                                                                                                                                         MAlonzo.Code.Lang.d__'8593'_500
                                                                                                                                                                                                                         v1
                                                                                                                                                                                                                         (coe
                                                                                                                                                                                                                            MAlonzo.Code.Lang.C_ar_34
                                                                                                                                                                                                                            (coe
                                                                                                                                                                                                                               MAlonzo.Code.Ar.d__'8855'__54
                                                                                                                                                                                                                               ()
                                                                                                                                                                                                                               erased
                                                                                                                                                                                                                               v37
                                                                                                                                                                                                                               v39))
                                                                                                                                                                                                                         (coe
                                                                                                                                                                                                                            MAlonzo.Code.Lang.C_ix_32
                                                                                                                                                                                                                            (coe
                                                                                                                                                                                                                               v37))
                                                                                                                                                                                                                         v12)
                                                                                                                                                                                                                      (coe
                                                                                                                                                                                                                         MAlonzo.Code.Lang.C_var_216
                                                                                                                                                                                                                         (coe
                                                                                                                                                                                                                            MAlonzo.Code.Lang.C_here_60)))
                                                                                                                                                                                                                   v43))
                                                                                                                                                                                                             erased
                                                                                                                                                                                                      _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                                                               _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                                                        _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                                                 _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                                          _ -> coe
                                                                                                                                                                                 v33
                                                                                                                                                                   _ -> coe
                                                                                                                                                                          v33
                                                                                                                                                            _ -> MAlonzo.RTE.mazUnreachableError) in
                                                                                                                                               coe
                                                                                                                                                 (case coe
                                                                                                                                                         v31 of
                                                                                                                                                    MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                                                                                                                                      -> case coe
                                                                                                                                                                v32 of
                                                                                                                                                           MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v34
                                                                                                                                                             -> case coe
                                                                                                                                                                       v34 of
                                                                                                                                                                  MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v35 v36
                                                                                                                                                                    -> case coe
                                                                                                                                                                              v36 of
                                                                                                                                                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v37 v38
                                                                                                                                                                           -> case coe
                                                                                                                                                                                     v38 of
                                                                                                                                                                                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v39 v40
                                                                                                                                                                                  -> case coe
                                                                                                                                                                                            v40 of
                                                                                                                                                                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v41 v42
                                                                                                                                                                                         -> coe
                                                                                                                                                                                              MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                                                                                                              (coe
                                                                                                                                                                                                 MAlonzo.Code.Lang.C_imap_226
                                                                                                                                                                                                 v35
                                                                                                                                                                                                 v37
                                                                                                                                                                                                 (coe
                                                                                                                                                                                                    MAlonzo.Code.Lang.C_bin_242
                                                                                                                                                                                                    v6
                                                                                                                                                                                                    v41
                                                                                                                                                                                                    (coe
                                                                                                                                                                                                       MAlonzo.Code.Lang.C_sel_228
                                                                                                                                                                                                       v35
                                                                                                                                                                                                       (coe
                                                                                                                                                                                                          MAlonzo.Code.Lang.d__'8593'_500
                                                                                                                                                                                                          v1
                                                                                                                                                                                                          (coe
                                                                                                                                                                                                             MAlonzo.Code.Lang.C_ar_34
                                                                                                                                                                                                             (coe
                                                                                                                                                                                                                MAlonzo.Code.Ar.d__'8855'__54
                                                                                                                                                                                                                ()
                                                                                                                                                                                                                erased
                                                                                                                                                                                                                v35
                                                                                                                                                                                                                v37))
                                                                                                                                                                                                          (coe
                                                                                                                                                                                                             MAlonzo.Code.Lang.C_ix_32
                                                                                                                                                                                                             (coe
                                                                                                                                                                                                                v35))
                                                                                                                                                                                                          v14)
                                                                                                                                                                                                       (coe
                                                                                                                                                                                                          MAlonzo.Code.Lang.C_var_216
                                                                                                                                                                                                          (coe
                                                                                                                                                                                                             MAlonzo.Code.Lang.C_here_60)))))
                                                                                                                                                                                              erased
                                                                                                                                                                                       _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                                                _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                                         _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                                  _ -> MAlonzo.RTE.mazUnreachableError
                                                                                                                                                           _ -> coe
                                                                                                                                                                  v33
                                                                                                                                                    _ -> coe
                                                                                                                                                           v33)
                                                                                                                                          _ -> MAlonzo.RTE.mazUnreachableError))
                                                                                                                             _ -> MAlonzo.RTE.mazUnreachableError)
                                                                                                              _ -> MAlonzo.RTE.mazUnreachableError))
                                                                                          _ -> MAlonzo.RTE.mazUnreachableError))
                                                                      _ -> MAlonzo.RTE.mazUnreachableError))
                                                  _ -> MAlonzo.RTE.mazUnreachableError)
                                        _ -> MAlonzo.RTE.mazUnreachableError
                                 _ -> MAlonzo.RTE.mazUnreachableError))
                    MAlonzo.Code.Lang.C_mul_192
                      -> let v10 = coe du_opt_214 (coe v0) (coe v1) (coe v2) (coe v7) in
                         coe
                           (let v11 = coe du_opt_214 (coe v0) (coe v1) (coe v2) (coe v8) in
                            coe
                              (case coe v10 of
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                                   -> let v14
                                            = case coe v11 of
                                                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                                  -> let v16
                                                           = coe
                                                               MAlonzo.Code.LangEq.du_isImaps_404
                                                               (coe v14) in
                                                     coe
                                                       (case coe v16 of
                                                          MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v17 v18
                                                            -> if coe v17
                                                                 then case coe v18 of
                                                                        MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v19
                                                                          -> case coe v19 of
                                                                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v20 v21
                                                                                 -> coe
                                                                                      MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                      (coe
                                                                                         MAlonzo.Code.Lang.C_imaps_222
                                                                                         (coe
                                                                                            MAlonzo.Code.Lang.C_bin_242
                                                                                            v6
                                                                                            (coe
                                                                                               MAlonzo.Code.Lang.C_sels_224
                                                                                               v9
                                                                                               (coe
                                                                                                  MAlonzo.Code.Lang.d__'8593'_500
                                                                                                  v1
                                                                                                  v2
                                                                                                  (coe
                                                                                                     MAlonzo.Code.Lang.C_ix_32
                                                                                                     (coe
                                                                                                        v9))
                                                                                                  v12)
                                                                                               (coe
                                                                                                  MAlonzo.Code.Lang.C_var_216
                                                                                                  (coe
                                                                                                     MAlonzo.Code.Lang.C_here_60)))
                                                                                            v20))
                                                                                      erased
                                                                               _ -> MAlonzo.RTE.mazUnreachableError
                                                                        _ -> MAlonzo.RTE.mazUnreachableError
                                                                 else coe
                                                                        seq (coe v18)
                                                                        (let v19
                                                                               = coe
                                                                                   MAlonzo.Code.LangEq.du_isImaps_404
                                                                                   (coe v12) in
                                                                         coe
                                                                           (case coe v19 of
                                                                              MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v20 v21
                                                                                -> if coe v20
                                                                                     then case coe
                                                                                                 v21 of
                                                                                            MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v22
                                                                                              -> case coe
                                                                                                        v22 of
                                                                                                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v23 v24
                                                                                                     -> coe
                                                                                                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                          (coe
                                                                                                             MAlonzo.Code.Lang.C_imaps_222
                                                                                                             (coe
                                                                                                                MAlonzo.Code.Lang.C_bin_242
                                                                                                                v6
                                                                                                                v23
                                                                                                                (coe
                                                                                                                   MAlonzo.Code.Lang.C_sels_224
                                                                                                                   v9
                                                                                                                   (coe
                                                                                                                      MAlonzo.Code.Lang.d__'8593'_500
                                                                                                                      v1
                                                                                                                      v2
                                                                                                                      (coe
                                                                                                                         MAlonzo.Code.Lang.C_ix_32
                                                                                                                         (coe
                                                                                                                            v9))
                                                                                                                      v14)
                                                                                                                   (coe
                                                                                                                      MAlonzo.Code.Lang.C_var_216
                                                                                                                      (coe
                                                                                                                         MAlonzo.Code.Lang.C_here_60)))))
                                                                                                          erased
                                                                                                   _ -> MAlonzo.RTE.mazUnreachableError
                                                                                            _ -> MAlonzo.RTE.mazUnreachableError
                                                                                     else coe
                                                                                            seq
                                                                                            (coe
                                                                                               v21)
                                                                                            (coe
                                                                                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                               (coe
                                                                                                  MAlonzo.Code.Lang.C_bin_242
                                                                                                  v6
                                                                                                  v12
                                                                                                  v14)
                                                                                               erased)
                                                                              _ -> MAlonzo.RTE.mazUnreachableError))
                                                          _ -> MAlonzo.RTE.mazUnreachableError)
                                                _ -> MAlonzo.RTE.mazUnreachableError in
                                      coe
                                        (case coe v12 of
                                           MAlonzo.Code.Lang.C_one_220
                                             -> case coe v11 of
                                                  MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v17 v18
                                                    -> coe
                                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                         (coe v17) erased
                                                  _ -> MAlonzo.RTE.mazUnreachableError
                                           MAlonzo.Code.Lang.C_zero'45'but_236 v16 v18 v19 v20
                                             -> case coe v11 of
                                                  MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v21 v22
                                                    -> coe
                                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                         (coe
                                                            MAlonzo.Code.Lang.C_zero'45'but_236 v16
                                                            v18 v19
                                                            (coe
                                                               MAlonzo.Code.Lang.C_bin_242 v6 v20
                                                               v21))
                                                         erased
                                                  _ -> MAlonzo.RTE.mazUnreachableError
                                           _ -> coe v14)
                                 _ -> MAlonzo.RTE.mazUnreachableError))
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_scaledown_244 v6 v7
        -> let v8 = coe du_opt_214 (coe v0) (coe v1) (coe v2) (coe v7) in
           coe
             (case coe v8 of
                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v9 v10
                  -> let v11
                           = coe
                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                               (coe MAlonzo.Code.Lang.C_scaledown_244 v6 v9) erased in
                     coe
                       (case coe v9 of
                          MAlonzo.Code.Lang.C_imaps_222 v14
                            -> coe
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                 (coe
                                    MAlonzo.Code.Lang.C_imaps_222
                                    (coe MAlonzo.Code.Lang.C_scaledown_244 v6 v14))
                                 erased
                          MAlonzo.Code.Lang.C_imap_226 v13 v14 v15
                            -> coe
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                 (coe
                                    MAlonzo.Code.Lang.C_imap_226 v13 v14
                                    (coe MAlonzo.Code.Lang.C_scaledown_244 v6 v15))
                                 erased
                          MAlonzo.Code.Lang.C_zero'45'but_236 v13 v15 v16 v17
                            -> coe du_foo_2030 (coe v13) (coe v15) (coe v16) (coe v17) (coe v6)
                          _ -> coe v11)
                _ -> MAlonzo.RTE.mazUnreachableError)
      MAlonzo.Code.Lang.C_let'8242'_246 v5 v7 v8
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v9
               -> let v10
                        = coe
                            du_opt_214 (coe v0) (coe v1)
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)) (coe v7) in
                  coe
                    (let v11
                           = coe
                               du_opt_214 (coe v0)
                               (coe
                                  MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                                  (coe MAlonzo.Code.Lang.C_ar_34 (coe v5)))
                               (coe v2) (coe v8) in
                     coe
                       (case coe v10 of
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                            -> case coe v11 of
                                 MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v14 v15
                                   -> let v16 = coe MAlonzo.Code.LangEq.du_isVar_72 (coe v12) in
                                      coe
                                        (case coe v16 of
                                           MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v17 v18
                                             -> if coe v17
                                                  then case coe v18 of
                                                         MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v19
                                                           -> case coe v19 of
                                                                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v20 v21
                                                                  -> coe
                                                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                       (coe
                                                                          MAlonzo.Code.Lang.d_sub_554
                                                                          (coe
                                                                             MAlonzo.Code.Lang.C__'9657'__40
                                                                             (coe v1)
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C_ar_34
                                                                                (coe v5)))
                                                                          (coe v2) (coe v1)
                                                                          (coe v14)
                                                                          (coe
                                                                             MAlonzo.Code.Lang.C__'9657'__516
                                                                             (MAlonzo.Code.Lang.d_sub'45'id_548
                                                                                (coe v1))
                                                                             (coe
                                                                                MAlonzo.Code.Lang.C_var_216
                                                                                v20)))
                                                                       erased
                                                                _ -> MAlonzo.RTE.mazUnreachableError
                                                         _ -> MAlonzo.RTE.mazUnreachableError
                                                  else coe
                                                         seq (coe v18)
                                                         (let v19
                                                                = coe
                                                                    MAlonzo.Code.LangEq.du_isLet_1704
                                                                    (coe v12) in
                                                          coe
                                                            (case coe v19 of
                                                               MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v20 v21
                                                                 -> let v22
                                                                          = coe
                                                                              MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                              (coe
                                                                                 MAlonzo.Code.Lang.C_let'8242'_246
                                                                                 v5 v12 v14)
                                                                              erased in
                                                                    coe
                                                                      (case coe v20 of
                                                                         MAlonzo.Code.Agda.Builtin.Bool.C_true_10
                                                                           -> case coe v21 of
                                                                                MAlonzo.Code.Relation.Nullary.Reflects.C_of'696'_22 v23
                                                                                  -> case coe v23 of
                                                                                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v24 v25
                                                                                         -> case coe
                                                                                                   v25 of
                                                                                              MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v26 v27
                                                                                                -> case coe
                                                                                                          v27 of
                                                                                                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v28 v29
                                                                                                       -> coe
                                                                                                            MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                                                                                            (coe
                                                                                                               MAlonzo.Code.Lang.C_let'8242'_246
                                                                                                               v24
                                                                                                               v26
                                                                                                               (coe
                                                                                                                  MAlonzo.Code.Lang.C_let'8242'_246
                                                                                                                  v5
                                                                                                                  v28
                                                                                                                  (MAlonzo.Code.Lang.d_wk_368
                                                                                                                     (coe
                                                                                                                        MAlonzo.Code.Lang.C__'9657'__40
                                                                                                                        (coe
                                                                                                                           v1)
                                                                                                                        (coe
                                                                                                                           MAlonzo.Code.Lang.C_ar_34
                                                                                                                           (coe
                                                                                                                              v5)))
                                                                                                                     (coe
                                                                                                                        MAlonzo.Code.Lang.C__'9657'__40
                                                                                                                        (coe
                                                                                                                           MAlonzo.Code.Lang.C__'9657'__40
                                                                                                                           (coe
                                                                                                                              v1)
                                                                                                                           (coe
                                                                                                                              MAlonzo.Code.Lang.C_ar_34
                                                                                                                              (coe
                                                                                                                                 v24)))
                                                                                                                        (coe
                                                                                                                           MAlonzo.Code.Lang.C_ar_34
                                                                                                                           (coe
                                                                                                                              v5)))
                                                                                                                     (coe
                                                                                                                        v2)
                                                                                                                     (coe
                                                                                                                        MAlonzo.Code.Lang.C_keep_354
                                                                                                                        (coe
                                                                                                                           MAlonzo.Code.Lang.C_skip_352
                                                                                                                           (MAlonzo.Code.Lang.d_'8838''45'eq_494
                                                                                                                              (coe
                                                                                                                                 v1))))
                                                                                                                     (coe
                                                                                                                        v14))))
                                                                                                            erased
                                                                                                     _ -> MAlonzo.RTE.mazUnreachableError
                                                                                              _ -> MAlonzo.RTE.mazUnreachableError
                                                                                       _ -> MAlonzo.RTE.mazUnreachableError
                                                                                _ -> coe v22
                                                                         _ -> coe v22)
                                                               _ -> MAlonzo.RTE.mazUnreachableError))
                                           _ -> MAlonzo.RTE.mazUnreachableError)
                                 _ -> MAlonzo.RTE.mazUnreachableError
                          _ -> MAlonzo.RTE.mazUnreachableError))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_un_248 v6 v7
        -> coe
             seq (coe v6)
             (let v8 = coe du_opt_214 (coe v0) (coe v1) (coe v2) (coe v7) in
              coe
                (case coe v8 of
                   MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v9 v10
                     -> coe
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                          (coe MAlonzo.Code.Lang.C_un_248 v6 v9) erased
                   _ -> MAlonzo.RTE.mazUnreachableError))
      MAlonzo.Code.Lang.C_maximum_250 v5 v7
        -> let v8
                 = coe
                     du_opt_214 (coe v0)
                     (coe
                        MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                        (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                     (coe v2) (coe v7) in
           coe
             (case coe v8 of
                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v9 v10
                  -> coe
                       MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                       (coe MAlonzo.Code.Lang.C_maximum_250 v5 v9) erased
                _ -> MAlonzo.RTE.mazUnreachableError)
      _ -> MAlonzo.RTE.mazUnreachableError
-- Opt._.go
d_go_378 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_go_378 = erased
-- Opt._.foo
d_foo_484 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_foo_484 = erased
-- Opt._.go
d_go_778 ::
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_go_778 = erased
-- Opt._.go
d_go_852 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_go_852 = erased
-- Opt._.foo
d_foo_922 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_foo_922 = erased
-- Opt._.foo
d_foo_1042 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_foo_1042 = erased
-- Opt._.go
d_go_1218 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_go_1218 = erased
-- Opt._.go
d_go_1376 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_go_1376 = erased
-- Opt._.go
d_go_1454 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  [Integer] ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Ar.T_Pointw'8322'_968 ->
  MAlonzo.Code.Ar.T_Pointw'8323'_990 ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_go_1454 = erased
-- Opt._.foo
d_foo_1738 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  MAlonzo.Code.Lang.T__'8712'__58 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20 ->
  MAlonzo.Code.Relation.Nullary.Decidable.Core.T_Dec_20 ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_foo_1738 = erased
-- Opt._.foo
d_foo_1846 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_foo_1846 = erased
-- Opt._.foo
d_foo_2030 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  Integer -> MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14
d_foo_2030 ~v0 ~v1 ~v2 ~v3 ~v4 v5 v6 v7 v8 ~v9 v10
  = du_foo_2030 v5 v6 v7 v8 v10
du_foo_2030 ::
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  Integer -> MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14
du_foo_2030 v0 v1 v2 v3 v4
  = let v5
          = coe
              MAlonzo.Code.Relation.Nullary.Decidable.Core.du_map'8242'_178
              erased
              (\ v5 ->
                 coe
                   MAlonzo.Code.Data.Nat.Properties.du_'8801''8658''8801''7495'_2786
                   (coe v4))
              (coe
                 MAlonzo.Code.Relation.Nullary.Decidable.Core.d_T'63'_72
                 (coe eqInt (coe v4) (coe (0 :: Integer)))) in
    coe
      (case coe v5 of
         MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v6 v7
           -> if coe v6
                then coe
                       seq (coe v7)
                       (coe
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                          (coe
                             MAlonzo.Code.Lang.C_scaledown_244 v4
                             (coe MAlonzo.Code.Lang.C_zero'45'but_236 v0 v1 v2 v3))
                          erased)
                else coe
                       seq (coe v7)
                       (coe
                          MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                          (coe
                             MAlonzo.Code.Lang.C_zero'45'but_236 v0 v1 v2
                             (coe MAlonzo.Code.Lang.C_scaledown_244 v4 v3))
                          erased)
         _ -> MAlonzo.RTE.mazUnreachableError)
-- Opt._._.foo'
d_foo''_2048 ::
  Integer ->
  (MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_foo''_2048 = erased
-- Opt._.foo
d_foo_2190 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.Irrelevant.T_Irrelevant_20) ->
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  (AgdaAny ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  MAlonzo.Code.Lang.T_E_214 ->
  (MAlonzo.Code.Agda.Builtin.Sigma.T_Σ_14 ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_foo_2190 = erased
