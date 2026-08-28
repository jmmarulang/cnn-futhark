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

module MAlonzo.Code.Eval where

import MAlonzo.RTE (coe, erased, AgdaAny, addInt, subInt, mulInt,
                    quotInt, remInt, geqInt, ltInt, eqInt, add64, sub64, mul64, quot64,
                    rem64, lt64, eq64, word64FromNat, word64ToNat)
import qualified MAlonzo.RTE
import qualified Data.Text
import qualified MAlonzo.Code.Agda.Builtin.Equality
import qualified MAlonzo.Code.Agda.Builtin.Sigma
import qualified MAlonzo.Code.Agda.Builtin.Unit
import qualified MAlonzo.Code.Ar
import qualified MAlonzo.Code.Data.List.Relation.Unary.All
import qualified MAlonzo.Code.Lang
import qualified MAlonzo.Code.Real
import qualified MAlonzo.Code.Relation.Nullary.Decidable.Core

-- Eval._._+_
d__'43'__10 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 -> AgdaAny -> AgdaAny -> AgdaAny
d__'43'__10 v0 ~v1 = du__'43'__10 v0
du__'43'__10 ::
  MAlonzo.Code.Real.T_Real_2 -> AgdaAny -> AgdaAny -> AgdaAny
du__'43'__10 v0 = coe MAlonzo.Code.Real.d__'43'__34 (coe v0)
-- Eval._.R
d_R_26 ::
  MAlonzo.Code.Real.T_Real_2 -> MAlonzo.Code.Real.T_RealProp_68 -> ()
d_R_26 = erased
-- Eval._.fromℕ
d_fromℕ_30 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 -> Integer -> AgdaAny
d_fromℕ_30 v0 ~v1 = du_fromℕ_30 v0
du_fromℕ_30 :: MAlonzo.Code.Real.T_Real_2 -> Integer -> AgdaAny
du_fromℕ_30 v0 = coe MAlonzo.Code.Real.d_fromℕ_30 (coe v0)
-- Eval.⟦_⟧ˢ
d_'10214'_'10215''738'_72 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 -> MAlonzo.Code.Lang.T_IS_30 -> ()
d_'10214'_'10215''738'_72 = erased
-- Eval.⟦_⟧ᶜ
d_'10214'_'10215''7580'_78 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 -> MAlonzo.Code.Lang.T_Ctx_36 -> ()
d_'10214'_'10215''7580'_78 = erased
-- Eval.lookup
d_lookup_84 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8712'__58 -> AgdaAny -> AgdaAny
d_lookup_84 ~v0 ~v1 ~v2 v3 v4 v5 = du_lookup_84 v3 v4 v5
du_lookup_84 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8712'__58 -> AgdaAny -> AgdaAny
du_lookup_84 v0 v1 v2
  = case coe v1 of
      MAlonzo.Code.Lang.C_here_60
        -> case coe v2 of
             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v5 v6 -> coe v6
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_there_62 v6
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v7 v8
               -> case coe v2 of
                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v9 v10
                      -> coe du_lookup_84 (coe v7) (coe v6) (coe v9)
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.zbs
d_zbs_96 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  AgdaAny
d_zbs_96 v0 ~v1 v2 v3 v4 v5 = du_zbs_96 v0 v2 v3 v4 v5
du_zbs_96 ::
  MAlonzo.Code.Real.T_Real_2 ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  AgdaAny
du_zbs_96 v0 v1 v2 v3 v4
  = let v5
          = MAlonzo.Code.Ar.d__'8799''8346'__646
              (coe v1) (coe v2) (coe v3) in
    coe
      (case coe v5 of
         MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v6 v7
           -> if coe v6
                then coe seq (coe v7) (coe v4 v2)
                else coe
                       seq (coe v7) (coe MAlonzo.Code.Real.d_fromℕ_30 v0 (0 :: Integer))
         _ -> MAlonzo.RTE.mazUnreachableError)
-- Eval.zb
d_zb_124 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny
d_zb_124 v0 ~v1 v2 ~v3 v4 v5 v6 = du_zb_124 v0 v2 v4 v5 v6
du_zb_124 ::
  MAlonzo.Code.Real.T_Real_2 ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny
du_zb_124 v0 v1 v2 v3 v4
  = let v5
          = MAlonzo.Code.Ar.d__'8799''8346'__646
              (coe v1) (coe v2) (coe v3) in
    coe
      (case coe v5 of
         MAlonzo.Code.Relation.Nullary.Decidable.Core.C__because__32 v6 v7
           -> if coe v6
                then coe seq (coe v7) (coe v4)
                else coe
                       seq (coe v7)
                       (coe (\ v8 -> coe MAlonzo.Code.Real.d_fromℕ_30 v0 (0 :: Integer)))
         _ -> MAlonzo.RTE.mazUnreachableError)
-- Eval.Softmax
d_Softmax_148 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny
d_Softmax_148 v0 ~v1 v2 v3 = du_Softmax_148 v0 v2 v3
du_Softmax_148 ::
  MAlonzo.Code.Real.T_Real_2 ->
  [Integer] ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny
du_Softmax_148 v0 v1 v2
  = coe
      MAlonzo.Code.Ar.du_map_146
      (coe
         (\ v3 ->
            coe
              MAlonzo.Code.Real.d__'247'__40 v0 v3
              (coe du_total_160 (coe v0) (coe v1) (coe v2))))
      (coe du_exps_158 (coe v0) (coe v2))
-- Eval._.exps
d_exps_158 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny
d_exps_158 v0 ~v1 ~v2 v3 = du_exps_158 v0 v3
du_exps_158 ::
  MAlonzo.Code.Real.T_Real_2 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny
du_exps_158 v0 v1
  = coe
      MAlonzo.Code.Ar.du_map_146
      (coe MAlonzo.Code.Real.d_e'94'__44 (coe v0)) (coe v1)
-- Eval._.total
d_total_160 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  AgdaAny
d_total_160 v0 ~v1 v2 v3 = du_total_160 v0 v2 v3
du_total_160 ::
  MAlonzo.Code.Real.T_Real_2 ->
  [Integer] ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  AgdaAny
du_total_160 v0 v1 v2
  = coe
      MAlonzo.Code.Ar.du_sum_326 (coe v1)
      (coe MAlonzo.Code.Real.d__'43'__34 (coe v0))
      (coe MAlonzo.Code.Real.d_fromℕ_30 v0 (0 :: Integer))
      (coe du_exps_158 (coe v0) (coe v2))
-- Eval.eval
d_eval_164 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> AgdaAny -> AgdaAny
d_eval_164 v0 ~v1 v2 v3 v4 v5 = du_eval_164 v0 v2 v3 v4 v5
du_eval_164 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> AgdaAny -> AgdaAny
du_eval_164 v0 v1 v2 v3 v4
  = case coe v3 of
      MAlonzo.Code.Lang.C_var_216 v7
        -> coe du_lookup_84 (coe v1) (coe v7) (coe v4)
      MAlonzo.Code.Lang.C_zero_218
        -> let v7 = coe MAlonzo.Code.Real.d_fromℕ_30 v0 (0 :: Integer) in
           coe (coe (\ v8 -> v7))
      MAlonzo.Code.Lang.C_one_220
        -> let v7 = coe MAlonzo.Code.Real.d_fromℕ_30 v0 (1 :: Integer) in
           coe (coe (\ v8 -> v7))
      MAlonzo.Code.Lang.C_imaps_222 v7
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v8
               -> coe
                    (\ v9 ->
                       coe
                         du_eval_164 v0
                         (coe
                            MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                            (coe MAlonzo.Code.Lang.C_ix_32 (coe v8)))
                         (coe MAlonzo.Code.Lang.C_ar_34 (coe MAlonzo.Code.Lang.d_unit_212))
                         v7
                         (coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) (coe v9))
                         (coe MAlonzo.Code.Data.List.Relation.Unary.All.C_'91''93'_50))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sels_224 v6 v7 v8
        -> let v9
                 = coe
                     du_eval_164 v0 v1 (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) v7 v4
                     (coe
                        du_eval_164 (coe v0) (coe v1)
                        (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)) (coe v8) (coe v4)) in
           coe (coe (\ v10 -> v9))
      MAlonzo.Code.Lang.C_imap_226 v6 v7 v8
        -> coe
             (\ v9 ->
                coe
                  du_eval_164 v0
                  (coe
                     MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                     (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)))
                  (coe MAlonzo.Code.Lang.C_ar_34 (coe v7)) v8
                  (coe
                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                     (coe
                        MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28
                        (coe MAlonzo.Code.Ar.du_splitP_172 (coe v6) (coe v9))))
                  (MAlonzo.Code.Agda.Builtin.Sigma.d_snd_30
                     (coe MAlonzo.Code.Ar.du_splitP_172 (coe v6) (coe v9))))
      MAlonzo.Code.Lang.C_sel_228 v6 v8 v9
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v10
               -> coe
                    MAlonzo.Code.Ar.du_nest_164 (coe v6)
                    (coe
                       du_eval_164 (coe v0) (coe v1)
                       (coe
                          MAlonzo.Code.Lang.C_ar_34
                          (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v6 v10))
                       (coe v8) (coe v4))
                    (coe
                       du_eval_164 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)) (coe v9) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_imapb_230 v5 v6 v9 v10
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v11
               -> coe
                    MAlonzo.Code.Ar.du_imapb_1286 (coe v5) (coe v6) (coe v11)
                    (coe
                       (\ v12 ->
                          coe
                            du_eval_164 (coe v0)
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                               (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)))
                            (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe v10)
                            (coe
                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) (coe v12))))
                    (coe v9)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_selb_232 v5 v7 v9 v10 v11
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v12
               -> coe
                    MAlonzo.Code.Ar.du_selb_1276 (coe v7) (coe v5) (coe v12)
                    (coe
                       du_eval_164 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v7)) (coe v10) (coe v4))
                    (coe v9)
                    (coe
                       du_eval_164 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v5)) (coe v11) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sum_234 v6 v8
        -> coe
             MAlonzo.Code.Ar.du_sum_326 (coe v6)
             (coe
                MAlonzo.Code.Ar.du_zipWith_154
                (coe MAlonzo.Code.Real.d__'43'__34 (coe v0)))
             (let v9 = coe MAlonzo.Code.Real.d_fromℕ_30 v0 (0 :: Integer) in
              coe (coe (\ v10 -> v9)))
             (coe
                (\ v9 ->
                   coe
                     du_eval_164 (coe v0)
                     (coe
                        MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                        (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)))
                     (coe v2) (coe v8)
                     (coe
                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) (coe v9))))
      MAlonzo.Code.Lang.C_zero'45'but_236 v6 v8 v9 v10
        -> coe
             du_zb_124 (coe v0) (coe v6)
             (coe
                du_eval_164 (coe v0) (coe v1)
                (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)) (coe v8) (coe v4))
             (coe
                du_eval_164 (coe v0) (coe v1)
                (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)) (coe v9) (coe v4))
             (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v10) (coe v4))
      MAlonzo.Code.Lang.C_slide_238 v6 v7 v8 v10 v11 v12 v13
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v14
               -> coe
                    MAlonzo.Code.Ar.du_slide_1180 (coe v6) (coe v7) (coe v8) (coe v14)
                    (coe
                       du_eval_164 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)) (coe v10) (coe v4))
                    (coe v11)
                    (coe
                       du_eval_164 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v8)) (coe v12) (coe v4))
                    (coe v13)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_backslide_240 v6 v7 v8 v10 v11 v12 v13
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v14
               -> coe
                    MAlonzo.Code.Ar.du_backslide_1194 (coe v6) (coe v7) (coe v8)
                    (coe v14)
                    (coe
                       du_eval_164 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ix_32 (coe v6)) (coe v10) (coe v4))
                    (coe
                       du_eval_164 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ar_34 (coe v7)) (coe v11) (coe v4))
                    (coe v12) (coe MAlonzo.Code.Real.d_fromℕ_30 v0 (0 :: Integer))
                    (coe v13)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_bin_242 v7 v8 v9
        -> case coe v7 of
             MAlonzo.Code.Lang.C_plus_190
               -> coe
                    MAlonzo.Code.Ar.du_zipWith_154
                    (coe MAlonzo.Code.Real.d__'43'__34 (coe v0))
                    (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v9) (coe v4))
             MAlonzo.Code.Lang.C_mul_192
               -> coe
                    MAlonzo.Code.Ar.du_zipWith_154
                    (coe MAlonzo.Code.Real.d__'42'__36 (coe v0))
                    (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v9) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_scaledown_244 v7 v8
        -> coe
             MAlonzo.Code.Ar.du_map_146
             (coe
                (\ v9 ->
                   coe
                     MAlonzo.Code.Real.d__'247'__40 v0 v9
                     (coe MAlonzo.Code.Real.d_fromℕ_30 v0 v7)))
             (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
      MAlonzo.Code.Lang.C_let'8242'_246 v6 v8 v9
        -> coe
             du_eval_164 (coe v0)
             (coe
                MAlonzo.Code.Lang.C__'9657'__40 (coe v1)
                (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)))
             (coe v2) (coe v9)
             (coe
                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                (coe
                   du_eval_164 (coe v0) (coe v1)
                   (coe MAlonzo.Code.Lang.C_ar_34 (coe v6)) (coe v8) (coe v4)))
      MAlonzo.Code.Lang.C_un_248 v7 v8
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_34 v9
               -> case coe v7 of
                    MAlonzo.Code.Lang.C_logistic_196
                      -> coe
                           MAlonzo.Code.Ar.du_map_146
                           (coe MAlonzo.Code.Real.d_logistic'691'_56 (coe v0))
                           (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    MAlonzo.Code.Lang.C_neg_198
                      -> coe
                           MAlonzo.Code.Ar.du_map_146
                           (coe MAlonzo.Code.Real.d_'45'__42 (coe v0))
                           (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    MAlonzo.Code.Lang.C_rectifier_200
                      -> coe
                           MAlonzo.Code.Ar.du_map_146
                           (coe
                              MAlonzo.Code.Real.d__'8744'__38 v0
                              (MAlonzo.Code.Real.d_0'7523'_52 (coe v0)))
                           (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    MAlonzo.Code.Lang.C_squared_202
                      -> coe
                           MAlonzo.Code.Ar.du_map_146
                           (coe MAlonzo.Code.Real.d_'8730'__46 (coe v0))
                           (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    MAlonzo.Code.Lang.C_inverse_204
                      -> coe
                           MAlonzo.Code.Ar.du_map_146
                           (coe MAlonzo.Code.Real.d_1'47'__60 (coe v0))
                           (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    MAlonzo.Code.Lang.C_ind'45'positive_206
                      -> coe
                           MAlonzo.Code.Ar.du_map_146
                           (coe MAlonzo.Code.Real.d_I'43'_48 (coe v0))
                           (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    MAlonzo.Code.Lang.C_logarithm_208
                      -> coe
                           MAlonzo.Code.Ar.du_map_146
                           (coe MAlonzo.Code.Real.d_log_50 (coe v0))
                           (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    MAlonzo.Code.Lang.C_softmax_210
                      -> coe
                           du_Softmax_148 (coe v0) (coe v9)
                           (coe du_eval_164 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval._≈ᵃ_
d__'8776''7491'__310 ::
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  () ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  ()
d__'8776''7491'__310 = erased
-- Eval._≈ˢ_
d__'8776''738'__322 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_IS_30 -> AgdaAny -> AgdaAny -> ()
d__'8776''738'__322 = erased
-- Eval._≈ᵉ_
d__'8776''7497'__336 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> MAlonzo.Code.Lang.T_E_214 -> ()
d__'8776''7497'__336 = erased
-- Eval.reflᵉ
d_refl'7497'_348 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 -> AgdaAny -> AgdaAny
d_refl'7497'_348 = erased
-- Eval.reflˢ
d_refl'738'_370 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_IS_30 -> AgdaAny -> AgdaAny
d_refl'738'_370 = erased
-- Eval.reflᵃ
d_refl'7491'_380 ::
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  () ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_refl'7491'_380 = erased
-- Eval.symˢ
d_sym'738'_388 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  AgdaAny -> AgdaAny -> AgdaAny -> AgdaAny
d_sym'738'_388 = erased
-- Eval._∙ᵃ_
d__'8729''7491'__412 ::
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  () ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d__'8729''7491'__412 = erased
-- Eval._∙ˢ_
d__'8729''738'__426 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  AgdaAny -> AgdaAny -> AgdaAny -> AgdaAny -> AgdaAny -> AgdaAny
d__'8729''738'__426 = erased
-- Eval._≈ᶜ_
d__'8776''7580'__440 a0 a1 a2 a3 a4 = ()
data T__'8776''7580'__440
  = C_ε_442 | C__'9657'__452 T__'8776''7580'__440
-- Eval.reflᶜ
d_refl'7580'_456 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> AgdaAny -> T__'8776''7580'__440
d_refl'7580'_456 ~v0 ~v1 v2 ~v3 = du_refl'7580'_456 v2
du_refl'7580'_456 ::
  MAlonzo.Code.Lang.T_Ctx_36 -> T__'8776''7580'__440
du_refl'7580'_456 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_ε_38 -> coe C_ε_442
      MAlonzo.Code.Lang.C__'9657'__40 v1 v2
        -> coe C__'9657'__452 (coe du_refl'7580'_456 (coe v1))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval._∙ᶜ_
d__'8729''7580'__468 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  AgdaAny ->
  AgdaAny ->
  AgdaAny ->
  T__'8776''7580'__440 ->
  T__'8776''7580'__440 -> T__'8776''7580'__440
d__'8729''7580'__468 ~v0 ~v1 v2 v3 v4 v5 v6 v7
  = du__'8729''7580'__468 v2 v3 v4 v5 v6 v7
du__'8729''7580'__468 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  AgdaAny ->
  AgdaAny ->
  AgdaAny ->
  T__'8776''7580'__440 ->
  T__'8776''7580'__440 -> T__'8776''7580'__440
du__'8729''7580'__468 v0 v1 v2 v3 v4 v5
  = case coe v0 of
      MAlonzo.Code.Lang.C_ε_38
        -> coe seq (coe v4) (coe seq (coe v5) (coe C_ε_442))
      MAlonzo.Code.Lang.C__'9657'__40 v6 v7
        -> case coe v4 of
             C__'9657'__452 v12
               -> case coe v1 of
                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                      -> case coe v2 of
                           MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v18 v19
                             -> case coe v5 of
                                  C__'9657'__452 v24
                                    -> case coe v3 of
                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v28 v29
                                           -> coe
                                                C__'9657'__452
                                                (coe
                                                   du__'8729''7580'__468 (coe v6) (coe v16)
                                                   (coe v18) (coe v28) (coe v12) (coe v24))
                                         _ -> MAlonzo.RTE.mazUnreachableError
                                  _ -> MAlonzo.RTE.mazUnreachableError
                           _ -> MAlonzo.RTE.mazUnreachableError
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.lookup-≈ᶜ
d_lookup'45''8776''7580'_488 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  AgdaAny ->
  AgdaAny ->
  T__'8776''7580'__440 -> MAlonzo.Code.Lang.T__'8712'__58 -> AgdaAny
d_lookup'45''8776''7580'_488 = erased
-- Eval.eval-cong
d_eval'45'cong_506 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_E_214 ->
  AgdaAny -> AgdaAny -> T__'8776''7580'__440 -> AgdaAny
d_eval'45'cong_506 = erased
-- Eval.sub-env
d_sub'45'env_822 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Sub_522 -> AgdaAny -> AgdaAny
d_sub'45'env_822 v0 ~v1 v2 v3 v4 v5
  = du_sub'45'env_822 v0 v2 v3 v4 v5
du_sub'45'env_822 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Sub_522 -> AgdaAny -> AgdaAny
du_sub'45'env_822 v0 v1 v2 v3 v4
  = case coe v3 of
      MAlonzo.Code.Lang.C_ε_526
        -> coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8
      MAlonzo.Code.Lang.C__'9657'__528 v7 v8
        -> case coe v2 of
             MAlonzo.Code.Lang.C__'9657'__40 v9 v10
               -> coe
                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                    (coe
                       du_sub'45'env_822 (coe v0) (coe v1) (coe v9) (coe v7) (coe v4))
                    (coe du_eval_164 (coe v0) (coe v1) (coe v10) (coe v8) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.wk-env
d_wk'45'env_834 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8838'__352 -> AgdaAny -> AgdaAny
d_wk'45'env_834 ~v0 ~v1 v2 v3 v4 v5 = du_wk'45'env_834 v2 v3 v4 v5
du_wk'45'env_834 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T__'8838'__352 -> AgdaAny -> AgdaAny
du_wk'45'env_834 v0 v1 v2 v3
  = case coe v2 of
      MAlonzo.Code.Lang.C_ε_354 -> coe v3
      MAlonzo.Code.Lang.C_skip_356 v7
        -> case coe v1 of
             MAlonzo.Code.Lang.C__'9657'__40 v8 v9
               -> case coe v3 of
                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v10 v11
                      -> coe du_wk'45'env_834 (coe v0) (coe v8) (coe v7) (coe v10)
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_keep_358 v7
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v8 v9
               -> case coe v1 of
                    MAlonzo.Code.Lang.C__'9657'__40 v10 v11
                      -> case coe v3 of
                           MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                             -> coe
                                  MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                  (coe du_wk'45'env_834 (coe v8) (coe v10) (coe v7) (coe v12))
                                  (coe v13)
                           _ -> MAlonzo.RTE.mazUnreachableError
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.wk-env-id
d_wk'45'env'45'id_850 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> AgdaAny -> T__'8776''7580'__440
d_wk'45'env'45'id_850 ~v0 ~v1 v2 ~v3 = du_wk'45'env'45'id_850 v2
du_wk'45'env'45'id_850 ::
  MAlonzo.Code.Lang.T_Ctx_36 -> T__'8776''7580'__440
du_wk'45'env'45'id_850 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_ε_38 -> coe C_ε_442
      MAlonzo.Code.Lang.C__'9657'__40 v1 v2
        -> coe C__'9657'__452 (coe du_wk'45'env'45'id_850 (coe v1))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.eval-wkv
d_eval'45'wkv_862 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T__'8838'__352 ->
  MAlonzo.Code.Lang.T__'8712'__58 -> AgdaAny -> AgdaAny
d_eval'45'wkv_862 = erased
-- Eval.eval-wk
d_eval'45'wk_886 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T__'8838'__352 ->
  MAlonzo.Code.Lang.T_E_214 -> AgdaAny -> AgdaAny
d_eval'45'wk_886 = erased
-- Eval.sub-env-wks
d_sub'45'env'45'wks_1184 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Sub_522 ->
  MAlonzo.Code.Lang.T__'8838'__352 -> AgdaAny -> T__'8776''7580'__440
d_sub'45'env'45'wks_1184 ~v0 ~v1 ~v2 v3 ~v4 v5 ~v6 ~v7
  = du_sub'45'env'45'wks_1184 v3 v5
du_sub'45'env'45'wks_1184 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Sub_522 -> T__'8776''7580'__440
du_sub'45'env'45'wks_1184 v0 v1
  = case coe v1 of
      MAlonzo.Code.Lang.C_ε_526 -> coe C_ε_442
      MAlonzo.Code.Lang.C__'9657'__528 v4 v5
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v6 v7
               -> coe
                    C__'9657'__452 (coe du_sub'45'env'45'wks_1184 (coe v6) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.sub-env-cong
d_sub'45'env'45'cong_1202 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Sub_522 ->
  AgdaAny -> AgdaAny -> T__'8776''7580'__440 -> T__'8776''7580'__440
d_sub'45'env'45'cong_1202 ~v0 ~v1 ~v2 v3 v4 ~v5 ~v6 ~v7
  = du_sub'45'env'45'cong_1202 v3 v4
du_sub'45'env'45'cong_1202 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Sub_522 -> T__'8776''7580'__440
du_sub'45'env'45'cong_1202 v0 v1
  = case coe v1 of
      MAlonzo.Code.Lang.C_ε_526 -> coe C_ε_442
      MAlonzo.Code.Lang.C__'9657'__528 v4 v5
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v6 v7
               -> coe
                    C__'9657'__452 (coe du_sub'45'env'45'cong_1202 (coe v6) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.sub-env-sdrop
d_sub'45'env'45'sdrop_1218 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  AgdaAny ->
  AgdaAny -> MAlonzo.Code.Lang.T_Sub_522 -> T__'8776''7580'__440
d_sub'45'env'45'sdrop_1218 ~v0 ~v1 ~v2 ~v3 v4 ~v5 ~v6 v7
  = du_sub'45'env'45'sdrop_1218 v4 v7
du_sub'45'env'45'sdrop_1218 ::
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Sub_522 -> T__'8776''7580'__440
du_sub'45'env'45'sdrop_1218 v0 v1
  = case coe v1 of
      MAlonzo.Code.Lang.C_ε_526 -> coe C_ε_442
      MAlonzo.Code.Lang.C__'9657'__528 v4 v5
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__40 v6 v7
               -> coe
                    C__'9657'__452 (coe du_sub'45'env'45'sdrop_1218 (coe v6) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.sub-env-id
d_sub'45'env'45'id_1230 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> AgdaAny -> T__'8776''7580'__440
d_sub'45'env'45'id_1230 v0 ~v1 v2 v3
  = du_sub'45'env'45'id_1230 v0 v2 v3
du_sub'45'env'45'id_1230 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_36 -> AgdaAny -> T__'8776''7580'__440
du_sub'45'env'45'id_1230 v0 v1 v2
  = case coe v1 of
      MAlonzo.Code.Lang.C_ε_38 -> coe C_ε_442
      MAlonzo.Code.Lang.C__'9657'__40 v3 v4
        -> coe
             C__'9657'__452
             (coe
                du__'8729''7580'__468 (coe v3)
                (coe
                   du_sub'45'env_822 (coe v0) (coe v1) (coe v3)
                   (coe
                      MAlonzo.Code.Lang.d_sdrop_540 (coe v3) (coe v3) (coe v4)
                      (coe MAlonzo.Code.Lang.d_sub'45'id_560 (coe v3)))
                   (coe v2))
                (coe
                   du_sub'45'env_822 (coe v0) (coe v3) (coe v3)
                   (coe MAlonzo.Code.Lang.d_sub'45'id_560 (coe v3))
                   (coe MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28 (coe v2)))
                (coe MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28 (coe v2))
                (coe
                   du_sub'45'env'45'sdrop_1218 (coe v3)
                   (coe MAlonzo.Code.Lang.d_sub'45'id_560 (coe v3)))
                (coe
                   du_sub'45'env'45'id_1230 (coe v0) (coe v3)
                   (coe MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28 (coe v2))))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.eval-subv
d_eval'45'subv_1242 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_Sub_522 ->
  AgdaAny -> MAlonzo.Code.Lang.T__'8712'__58 -> AgdaAny
d_eval'45'subv_1242 = erased
-- Eval.eval-sub
d_eval'45'sub_1262 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_IS_30 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  MAlonzo.Code.Lang.T_E_214 ->
  AgdaAny -> MAlonzo.Code.Lang.T_Sub_522 -> AgdaAny
d_eval'45'sub_1262 = erased
-- Eval.eval-zb
d_eval'45'zb_1710 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  MAlonzo.Code.Lang.T_Ctx_36 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_214 ->
  MAlonzo.Code.Lang.T_E_214 ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_eval'45'zb_1710 = erased
-- Eval.ZeroBut.zbs-suc-r
d_zbs'45'suc'45'r_1790 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'suc'45'r_1790 = erased
-- Eval.ZeroBut.sum₁-zero
d_sum'8321''45'zero_1894 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_sum'8321''45'zero_1894 = erased
-- Eval.ZeroBut.sum-zero
d_sum'45'zero_1900 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_sum'45'zero_1900 = erased
-- Eval.ZeroBut.zbs-zero
d_zbs'45'zero_1912 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'zero_1912 = erased
-- Eval.ZeroBut._.go
d_go_1924 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_go_1924 = erased
-- Eval.ZeroBut.zbs-suc
d_zbs'45'suc_1936 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'suc_1936 = erased
-- Eval.ZeroBut.zbs-sum₁-s
d_zbs'45'sum'8321''45's_1982 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'sum'8321''45's_1982 = erased
-- Eval.ZeroBut.zbs-sum-s
d_zbs'45'sum'45's_2004 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'sum'45's_2004 = erased
-- Eval.ZeroBut.zb-zbs
d_zb'45'zbs_2040 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zb'45'zbs_2040 = erased
-- Eval.ZeroBut.zbs-sym
d_zbs'45'sym_2076 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'sym_2076 = erased
-- Eval.ZeroBut.zb-sym
d_zb'45'sym_2124 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zb'45'sym_2124 = erased
-- Eval.ZeroBut.zbs-cong
d_zbs'45'cong_2184 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'cong_2184 = erased
-- Eval.ZeroBut.zb-sum
d_zb'45'sum_2226 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zb'45'sum_2226 = erased
-- Eval.ZeroBut.zbs-ext
d_zbs'45'ext_2254 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'ext_2254 = erased
-- Eval.ZeroBut.zb-zbs-k
d_zb'45'zbs'45'k_2294 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zb'45'zbs'45'k_2294 = erased
