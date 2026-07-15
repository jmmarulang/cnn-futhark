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

-- Eval.⟦_⟧ˢ
d_'10214'_'10215''738'_38 ::
  MAlonzo.Code.Real.T_Real_2 -> MAlonzo.Code.Lang.T_IS_6 -> ()
d_'10214'_'10215''738'_38 = erased
-- Eval.⟦_⟧ᶜ
d_'10214'_'10215''7580'_44 ::
  MAlonzo.Code.Real.T_Real_2 -> MAlonzo.Code.Lang.T_Ctx_12 -> ()
d_'10214'_'10215''7580'_44 = erased
-- Eval.lookup
d_lookup_50 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T__'8712'__34 -> AgdaAny -> AgdaAny
d_lookup_50 ~v0 ~v1 v2 v3 v4 = du_lookup_50 v2 v3 v4
du_lookup_50 ::
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T__'8712'__34 -> AgdaAny -> AgdaAny
du_lookup_50 v0 v1 v2
  = case coe v1 of
      MAlonzo.Code.Lang.C_here_36
        -> case coe v2 of
             MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v5 v6 -> coe v6
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_there_38 v6
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__16 v7 v8
               -> case coe v2 of
                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v9 v10
                      -> coe du_lookup_50 (coe v7) (coe v6) (coe v9)
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.zbs
d_zbs_62 ::
  MAlonzo.Code.Real.T_Real_2 ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  AgdaAny
d_zbs_62 v0 v1 v2 v3 v4
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
d_zb_90 ::
  MAlonzo.Code.Real.T_Real_2 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny
d_zb_90 v0 v1 ~v2 v3 v4 v5 = du_zb_90 v0 v1 v3 v4 v5
du_zb_90 ::
  MAlonzo.Code.Real.T_Real_2 ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny
du_zb_90 v0 v1 v2 v3 v4
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
-- Eval.eval
d_eval_114 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_E_182 -> AgdaAny -> AgdaAny
d_eval_114 v0 v1 v2 v3 v4
  = case coe v3 of
      MAlonzo.Code.Lang.C_var_184 v7
        -> coe du_lookup_50 (coe v1) (coe v7) (coe v4)
      MAlonzo.Code.Lang.C_zero_186
        -> let v7 = coe MAlonzo.Code.Real.d_fromℕ_30 v0 (0 :: Integer) in
           coe (coe (\ v8 -> v7))
      MAlonzo.Code.Lang.C_one_188
        -> let v7 = coe MAlonzo.Code.Real.d_fromℕ_30 v0 (1 :: Integer) in
           coe (coe (\ v8 -> v7))
      MAlonzo.Code.Lang.C_imaps_190 v7
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_10 v8
               -> coe
                    (\ v9 ->
                       coe
                         d_eval_114 v0
                         (coe
                            MAlonzo.Code.Lang.C__'9657'__16 (coe v1)
                            (coe MAlonzo.Code.Lang.C_ix_8 (coe v8)))
                         (coe MAlonzo.Code.Lang.C_ar_10 (coe MAlonzo.Code.Lang.d_unit_180))
                         v7
                         (coe MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) (coe v9))
                         (coe MAlonzo.Code.Data.List.Relation.Unary.All.C_'91''93'_50))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sels_192 v6 v7 v8
        -> let v9
                 = coe
                     d_eval_114 v0 v1 (coe MAlonzo.Code.Lang.C_ar_10 (coe v6)) v7 v4
                     (d_eval_114
                        (coe v0) (coe v1) (coe MAlonzo.Code.Lang.C_ix_8 (coe v6)) (coe v8)
                        (coe v4)) in
           coe (coe (\ v10 -> v9))
      MAlonzo.Code.Lang.C_imap_194 v6 v7 v8
        -> coe
             (\ v9 ->
                coe
                  d_eval_114 v0
                  (coe
                     MAlonzo.Code.Lang.C__'9657'__16 (coe v1)
                     (coe MAlonzo.Code.Lang.C_ix_8 (coe v6)))
                  (coe MAlonzo.Code.Lang.C_ar_10 (coe v7)) v8
                  (coe
                     MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                     (coe
                        MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28
                        (coe MAlonzo.Code.Ar.du_splitP_172 (coe v6) (coe v9))))
                  (MAlonzo.Code.Agda.Builtin.Sigma.d_snd_30
                     (coe MAlonzo.Code.Ar.du_splitP_172 (coe v6) (coe v9))))
      MAlonzo.Code.Lang.C_sel_196 v6 v8 v9
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_10 v10
               -> coe
                    MAlonzo.Code.Ar.du_nest_164 (coe v6)
                    (coe
                       d_eval_114 (coe v0) (coe v1)
                       (coe
                          MAlonzo.Code.Lang.C_ar_10
                          (coe MAlonzo.Code.Ar.d__'8855'__54 () erased v6 v10))
                       (coe v8) (coe v4))
                    (coe
                       d_eval_114 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ix_8 (coe v6)) (coe v9) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_imapb_198 v5 v6 v9 v10
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_10 v11
               -> coe
                    MAlonzo.Code.Ar.du_imapb_1286 (coe v5) (coe v6) (coe v11)
                    (coe
                       (\ v12 ->
                          d_eval_114
                            (coe v0)
                            (coe
                               MAlonzo.Code.Lang.C__'9657'__16 (coe v1)
                               (coe MAlonzo.Code.Lang.C_ix_8 (coe v5)))
                            (coe MAlonzo.Code.Lang.C_ar_10 (coe v6)) (coe v10)
                            (coe
                               MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) (coe v12))))
                    (coe v9)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_selb_200 v5 v7 v9 v10 v11
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_10 v12
               -> coe
                    MAlonzo.Code.Ar.du_selb_1276 (coe v7) (coe v5) (coe v12)
                    (coe
                       d_eval_114 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ar_10 (coe v7)) (coe v10) (coe v4))
                    (coe v9)
                    (coe
                       d_eval_114 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ix_8 (coe v5)) (coe v11) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_sum_202 v6 v8
        -> coe
             MAlonzo.Code.Ar.du_sum_326 (coe v6)
             (coe
                MAlonzo.Code.Ar.du_zipWith_154
                (coe MAlonzo.Code.Real.d__'43'__34 (coe v0)))
             (let v9 = coe MAlonzo.Code.Real.d_fromℕ_30 v0 (0 :: Integer) in
              coe (coe (\ v10 -> v9)))
             (coe
                (\ v9 ->
                   d_eval_114
                     (coe v0)
                     (coe
                        MAlonzo.Code.Lang.C__'9657'__16 (coe v1)
                        (coe MAlonzo.Code.Lang.C_ix_8 (coe v6)))
                     (coe v2) (coe v8)
                     (coe
                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) (coe v9))))
      MAlonzo.Code.Lang.C_zero'45'but_204 v6 v8 v9 v10
        -> coe
             du_zb_90 (coe v0) (coe v6)
             (coe
                d_eval_114 (coe v0) (coe v1)
                (coe MAlonzo.Code.Lang.C_ix_8 (coe v6)) (coe v8) (coe v4))
             (coe
                d_eval_114 (coe v0) (coe v1)
                (coe MAlonzo.Code.Lang.C_ix_8 (coe v6)) (coe v9) (coe v4))
             (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v10) (coe v4))
      MAlonzo.Code.Lang.C_slide_206 v6 v7 v8 v10 v11 v12 v13
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_10 v14
               -> coe
                    MAlonzo.Code.Ar.du_slide_1180 (coe v6) (coe v7) (coe v8) (coe v14)
                    (coe
                       d_eval_114 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ix_8 (coe v6)) (coe v10) (coe v4))
                    (coe v11)
                    (coe
                       d_eval_114 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ar_10 (coe v8)) (coe v12) (coe v4))
                    (coe v13)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_backslide_208 v6 v7 v8 v10 v11 v12 v13
        -> case coe v2 of
             MAlonzo.Code.Lang.C_ar_10 v14
               -> coe
                    MAlonzo.Code.Ar.du_backslide_1194 (coe v6) (coe v7) (coe v8)
                    (coe v14)
                    (coe
                       d_eval_114 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ix_8 (coe v6)) (coe v10) (coe v4))
                    (coe
                       d_eval_114 (coe v0) (coe v1)
                       (coe MAlonzo.Code.Lang.C_ar_10 (coe v7)) (coe v11) (coe v4))
                    (coe v12) (coe MAlonzo.Code.Real.d_fromℕ_30 v0 (0 :: Integer))
                    (coe v13)
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_bin_210 v7 v8 v9
        -> case coe v7 of
             MAlonzo.Code.Lang.C_plus_158
               -> coe
                    MAlonzo.Code.Ar.du_zipWith_154
                    (coe MAlonzo.Code.Real.d__'43'__34 (coe v0))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v9) (coe v4))
             MAlonzo.Code.Lang.C_mul_160
               -> coe
                    MAlonzo.Code.Ar.du_zipWith_154
                    (coe MAlonzo.Code.Real.d__'42'__36 (coe v0))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v9) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_scaledown_212 v7 v8
        -> coe
             MAlonzo.Code.Ar.du_map_146
             (coe
                (\ v9 ->
                   coe
                     MAlonzo.Code.Real.d__'247'__40 v0 v9
                     (coe MAlonzo.Code.Real.d_fromℕ_30 v0 v7)))
             (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
      MAlonzo.Code.Lang.C_let'8242'_214 v6 v8 v9
        -> coe
             d_eval_114 (coe v0)
             (coe
                MAlonzo.Code.Lang.C__'9657'__16 (coe v1)
                (coe MAlonzo.Code.Lang.C_ar_10 (coe v6)))
             (coe v2) (coe v9)
             (coe
                MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4)
                (coe
                   d_eval_114 (coe v0) (coe v1)
                   (coe MAlonzo.Code.Lang.C_ar_10 (coe v6)) (coe v8) (coe v4)))
      MAlonzo.Code.Lang.C_un_216 v7 v8
        -> case coe v7 of
             MAlonzo.Code.Lang.C_logistic_164
               -> coe
                    MAlonzo.Code.Ar.du_map_146
                    (coe MAlonzo.Code.Real.d_logistic'691'_56 (coe v0))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
             MAlonzo.Code.Lang.C_neg_166
               -> coe
                    MAlonzo.Code.Ar.du_map_146
                    (coe MAlonzo.Code.Real.d_'45'__42 (coe v0))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
             MAlonzo.Code.Lang.C_exp_168
               -> coe
                    MAlonzo.Code.Ar.du_map_146
                    (coe MAlonzo.Code.Real.d_e'94'__44 (coe v0))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
             MAlonzo.Code.Lang.C_rectifier_170
               -> coe
                    MAlonzo.Code.Ar.du_map_146
                    (coe
                       MAlonzo.Code.Real.d__'8744'__38 v0
                       (MAlonzo.Code.Real.d_0'7523'_52 (coe v0)))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
             MAlonzo.Code.Lang.C_squared_172
               -> coe
                    MAlonzo.Code.Ar.du_map_146
                    (coe MAlonzo.Code.Real.d_'8730'__46 (coe v0))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
             MAlonzo.Code.Lang.C_inverse_174
               -> coe
                    MAlonzo.Code.Ar.du_map_146
                    (coe MAlonzo.Code.Real.d_1'47'__60 (coe v0))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
             MAlonzo.Code.Lang.C_ind'45'positive_176
               -> coe
                    MAlonzo.Code.Ar.du_map_146
                    (coe MAlonzo.Code.Real.d_I'43'_48 (coe v0))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
             MAlonzo.Code.Lang.C_logarithm_178
               -> coe
                    MAlonzo.Code.Ar.du_map_146
                    (coe MAlonzo.Code.Real.d_log_50 (coe v0))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v2) (coe v8) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_maximum_218 v6 v8
        -> coe
             MAlonzo.Code.Ar.du_sum_326 (coe v6)
             (coe
                MAlonzo.Code.Ar.du_zipWith_154
                (coe MAlonzo.Code.Real.d__'8744'__38 (coe v0)))
             (let v9 = MAlonzo.Code.Real.d_'45''8734''7523'_54 (coe v0) in
              coe (coe (\ v10 -> v9)))
             (coe
                (\ v9 ->
                   d_eval_114
                     (coe v0)
                     (coe
                        MAlonzo.Code.Lang.C__'9657'__16 (coe v1)
                        (coe MAlonzo.Code.Lang.C_ix_8 (coe v6)))
                     (coe v2) (coe v8)
                     (coe
                        MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 (coe v4) (coe v9))))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval._≈ᵃ_
d__'8776''7491'__266 ::
  MAlonzo.Code.Real.T_Real_2 ->
  [Integer] ->
  () ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  ()
d__'8776''7491'__266 = erased
-- Eval._≈ˢ_
d__'8776''738'__278 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_IS_6 -> AgdaAny -> AgdaAny -> ()
d__'8776''738'__278 = erased
-- Eval._≈ᵉ_
d__'8776''7497'__292 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_E_182 -> MAlonzo.Code.Lang.T_E_182 -> ()
d__'8776''7497'__292 = erased
-- Eval.reflᵉ
d_refl'7497'_304 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_E_182 -> AgdaAny -> AgdaAny
d_refl'7497'_304 = erased
-- Eval.reflˢ
d_refl'738'_326 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_IS_6 -> AgdaAny -> AgdaAny
d_refl'738'_326 = erased
-- Eval.reflᵃ
d_refl'7491'_336 ::
  MAlonzo.Code.Real.T_Real_2 ->
  [Integer] ->
  () ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_refl'7491'_336 = erased
-- Eval._∙ᵃ_
d__'8729''7491'__346 ::
  MAlonzo.Code.Real.T_Real_2 ->
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
d__'8729''7491'__346 = erased
-- Eval._∙ˢ_
d__'8729''738'__360 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  AgdaAny -> AgdaAny -> AgdaAny -> AgdaAny -> AgdaAny -> AgdaAny
d__'8729''738'__360 = erased
-- Eval._≈ᶜ_
d__'8776''7580'__374 a0 a1 a2 a3 = ()
data T__'8776''7580'__374
  = C_ε_376 | C__'9657'__386 T__'8776''7580'__374
-- Eval.reflᶜ
d_refl'7580'_390 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 -> AgdaAny -> T__'8776''7580'__374
d_refl'7580'_390 ~v0 v1 ~v2 = du_refl'7580'_390 v1
du_refl'7580'_390 ::
  MAlonzo.Code.Lang.T_Ctx_12 -> T__'8776''7580'__374
du_refl'7580'_390 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_ε_14 -> coe C_ε_376
      MAlonzo.Code.Lang.C__'9657'__16 v1 v2
        -> coe C__'9657'__386 (coe du_refl'7580'_390 (coe v1))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval._∙ᶜ_
d__'8729''7580'__402 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  AgdaAny ->
  AgdaAny ->
  AgdaAny ->
  T__'8776''7580'__374 ->
  T__'8776''7580'__374 -> T__'8776''7580'__374
d__'8729''7580'__402 ~v0 v1 v2 v3 v4 v5 v6
  = du__'8729''7580'__402 v1 v2 v3 v4 v5 v6
du__'8729''7580'__402 ::
  MAlonzo.Code.Lang.T_Ctx_12 ->
  AgdaAny ->
  AgdaAny ->
  AgdaAny ->
  T__'8776''7580'__374 ->
  T__'8776''7580'__374 -> T__'8776''7580'__374
du__'8729''7580'__402 v0 v1 v2 v3 v4 v5
  = case coe v0 of
      MAlonzo.Code.Lang.C_ε_14
        -> coe seq (coe v4) (coe seq (coe v5) (coe C_ε_376))
      MAlonzo.Code.Lang.C__'9657'__16 v6 v7
        -> case coe v4 of
             C__'9657'__386 v12
               -> case coe v1 of
                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v16 v17
                      -> case coe v2 of
                           MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v18 v19
                             -> case coe v5 of
                                  C__'9657'__386 v24
                                    -> case coe v3 of
                                         MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v28 v29
                                           -> coe
                                                C__'9657'__386
                                                (coe
                                                   du__'8729''7580'__402 (coe v6) (coe v16)
                                                   (coe v18) (coe v28) (coe v12) (coe v24))
                                         _ -> MAlonzo.RTE.mazUnreachableError
                                  _ -> MAlonzo.RTE.mazUnreachableError
                           _ -> MAlonzo.RTE.mazUnreachableError
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.lookup-≈ᶜ
d_lookup'45''8776''7580'_422 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  AgdaAny ->
  AgdaAny ->
  T__'8776''7580'__374 -> MAlonzo.Code.Lang.T__'8712'__34 -> AgdaAny
d_lookup'45''8776''7580'_422 = erased
-- Eval.eval-cong
d_eval'45'cong_440 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_E_182 ->
  AgdaAny -> AgdaAny -> T__'8776''7580'__374 -> AgdaAny
d_eval'45'cong_440 = erased
-- Eval.sub-env
d_sub'45'env_770 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Sub_472 -> AgdaAny -> AgdaAny
d_sub'45'env_770 v0 v1 v2 v3 v4
  = case coe v3 of
      MAlonzo.Code.Lang.C_ε_476
        -> coe MAlonzo.Code.Agda.Builtin.Unit.C_tt_8
      MAlonzo.Code.Lang.C__'9657'__478 v7 v8
        -> case coe v2 of
             MAlonzo.Code.Lang.C__'9657'__16 v9 v10
               -> coe
                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                    (coe d_sub'45'env_770 (coe v0) (coe v1) (coe v9) (coe v7) (coe v4))
                    (coe d_eval_114 (coe v0) (coe v1) (coe v10) (coe v8) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.wk-env
d_wk'45'env_782 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T__'8838'__310 -> AgdaAny -> AgdaAny
d_wk'45'env_782 ~v0 v1 v2 v3 v4 = du_wk'45'env_782 v1 v2 v3 v4
du_wk'45'env_782 ::
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T__'8838'__310 -> AgdaAny -> AgdaAny
du_wk'45'env_782 v0 v1 v2 v3
  = case coe v2 of
      MAlonzo.Code.Lang.C_ε_312 -> coe v3
      MAlonzo.Code.Lang.C_skip_314 v7
        -> case coe v1 of
             MAlonzo.Code.Lang.C__'9657'__16 v8 v9
               -> case coe v3 of
                    MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v10 v11
                      -> coe du_wk'45'env_782 (coe v0) (coe v8) (coe v7) (coe v10)
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      MAlonzo.Code.Lang.C_keep_316 v7
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__16 v8 v9
               -> case coe v1 of
                    MAlonzo.Code.Lang.C__'9657'__16 v10 v11
                      -> case coe v3 of
                           MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32 v12 v13
                             -> coe
                                  MAlonzo.Code.Agda.Builtin.Sigma.C__'44'__32
                                  (coe du_wk'45'env_782 (coe v8) (coe v10) (coe v7) (coe v12))
                                  (coe v13)
                           _ -> MAlonzo.RTE.mazUnreachableError
                    _ -> MAlonzo.RTE.mazUnreachableError
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.wk-env-id
d_wk'45'env'45'id_798 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 -> AgdaAny -> T__'8776''7580'__374
d_wk'45'env'45'id_798 ~v0 v1 ~v2 = du_wk'45'env'45'id_798 v1
du_wk'45'env'45'id_798 ::
  MAlonzo.Code.Lang.T_Ctx_12 -> T__'8776''7580'__374
du_wk'45'env'45'id_798 v0
  = case coe v0 of
      MAlonzo.Code.Lang.C_ε_14 -> coe C_ε_376
      MAlonzo.Code.Lang.C__'9657'__16 v1 v2
        -> coe C__'9657'__386 (coe du_wk'45'env'45'id_798 (coe v1))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.eval-wkv
d_eval'45'wkv_810 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T__'8838'__310 ->
  MAlonzo.Code.Lang.T__'8712'__34 -> AgdaAny -> AgdaAny
d_eval'45'wkv_810 = erased
-- Eval.eval-wk
d_eval'45'wk_834 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T__'8838'__310 ->
  MAlonzo.Code.Lang.T_E_182 -> AgdaAny -> AgdaAny
d_eval'45'wk_834 = erased
-- Eval.sub-env-wks
d_sub'45'env'45'wks_1144 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Sub_472 ->
  MAlonzo.Code.Lang.T__'8838'__310 -> AgdaAny -> T__'8776''7580'__374
d_sub'45'env'45'wks_1144 ~v0 ~v1 v2 ~v3 v4 ~v5 ~v6
  = du_sub'45'env'45'wks_1144 v2 v4
du_sub'45'env'45'wks_1144 ::
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Sub_472 -> T__'8776''7580'__374
du_sub'45'env'45'wks_1144 v0 v1
  = case coe v1 of
      MAlonzo.Code.Lang.C_ε_476 -> coe C_ε_376
      MAlonzo.Code.Lang.C__'9657'__478 v4 v5
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__16 v6 v7
               -> coe
                    C__'9657'__386 (coe du_sub'45'env'45'wks_1144 (coe v6) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.sub-env-cong
d_sub'45'env'45'cong_1162 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Sub_472 ->
  AgdaAny -> AgdaAny -> T__'8776''7580'__374 -> T__'8776''7580'__374
d_sub'45'env'45'cong_1162 ~v0 ~v1 v2 v3 ~v4 ~v5 ~v6
  = du_sub'45'env'45'cong_1162 v2 v3
du_sub'45'env'45'cong_1162 ::
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Sub_472 -> T__'8776''7580'__374
du_sub'45'env'45'cong_1162 v0 v1
  = case coe v1 of
      MAlonzo.Code.Lang.C_ε_476 -> coe C_ε_376
      MAlonzo.Code.Lang.C__'9657'__478 v4 v5
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__16 v6 v7
               -> coe
                    C__'9657'__386 (coe du_sub'45'env'45'cong_1162 (coe v6) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.sub-env-sdrop
d_sub'45'env'45'sdrop_1178 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  AgdaAny ->
  AgdaAny -> MAlonzo.Code.Lang.T_Sub_472 -> T__'8776''7580'__374
d_sub'45'env'45'sdrop_1178 ~v0 ~v1 ~v2 v3 ~v4 ~v5 v6
  = du_sub'45'env'45'sdrop_1178 v3 v6
du_sub'45'env'45'sdrop_1178 ::
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Sub_472 -> T__'8776''7580'__374
du_sub'45'env'45'sdrop_1178 v0 v1
  = case coe v1 of
      MAlonzo.Code.Lang.C_ε_476 -> coe C_ε_376
      MAlonzo.Code.Lang.C__'9657'__478 v4 v5
        -> case coe v0 of
             MAlonzo.Code.Lang.C__'9657'__16 v6 v7
               -> coe
                    C__'9657'__386 (coe du_sub'45'env'45'sdrop_1178 (coe v6) (coe v4))
             _ -> MAlonzo.RTE.mazUnreachableError
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.sub-env-id
d_sub'45'env'45'id_1190 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 -> AgdaAny -> T__'8776''7580'__374
d_sub'45'env'45'id_1190 v0 v1 v2
  = case coe v1 of
      MAlonzo.Code.Lang.C_ε_14 -> coe C_ε_376
      MAlonzo.Code.Lang.C__'9657'__16 v3 v4
        -> coe
             C__'9657'__386
             (coe
                du__'8729''7580'__402 (coe v3)
                (coe
                   d_sub'45'env_770 (coe v0) (coe v1) (coe v3)
                   (coe
                      MAlonzo.Code.Lang.d_sdrop_490 (coe v3) (coe v3) (coe v4)
                      (coe MAlonzo.Code.Lang.d_sub'45'id_510 (coe v3)))
                   (coe v2))
                (coe
                   d_sub'45'env_770 (coe v0) (coe v3) (coe v3)
                   (coe MAlonzo.Code.Lang.d_sub'45'id_510 (coe v3))
                   (coe MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28 (coe v2)))
                (coe MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28 (coe v2))
                (coe
                   du_sub'45'env'45'sdrop_1178 (coe v3)
                   (coe MAlonzo.Code.Lang.d_sub'45'id_510 (coe v3)))
                (coe
                   d_sub'45'env'45'id_1190 (coe v0) (coe v3)
                   (coe MAlonzo.Code.Agda.Builtin.Sigma.d_fst_28 (coe v2))))
      _ -> MAlonzo.RTE.mazUnreachableError
-- Eval.eval-subv
d_eval'45'subv_1202 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_Sub_472 ->
  AgdaAny -> MAlonzo.Code.Lang.T__'8712'__34 -> AgdaAny
d_eval'45'subv_1202 = erased
-- Eval.eval-sub
d_eval'45'sub_1222 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_IS_6 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  MAlonzo.Code.Lang.T_E_182 ->
  AgdaAny -> MAlonzo.Code.Lang.T_Sub_472 -> AgdaAny
d_eval'45'sub_1222 = erased
-- Eval.eval-zb
d_eval'45'zb_1686 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Lang.T_Ctx_12 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Lang.T_E_182 ->
  MAlonzo.Code.Lang.T_E_182 ->
  AgdaAny ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_eval'45'zb_1686 = erased
-- Eval.ZeroBut.zbs-suc-r
d_zbs'45'suc'45'r_1750 ::
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
d_zbs'45'suc'45'r_1750 = erased
-- Eval.ZeroBut.sum₁-zero
d_sum'8321''45'zero_1854 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_sum'8321''45'zero_1854 = erased
-- Eval.ZeroBut.sum-zero
d_sum'45'zero_1860 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] -> MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_sum'45'zero_1860 = erased
-- Eval.ZeroBut.zbs-zero
d_zbs'45'zero_1872 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'zero_1872 = erased
-- Eval.ZeroBut._.go
d_go_1884 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_go_1884 = erased
-- Eval.ZeroBut.zbs-suc
d_zbs'45'suc_1896 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'suc_1896 = erased
-- Eval.ZeroBut.zbs-sum₁-s
d_zbs'45'sum'8321''45's_1942 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  Integer ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'sum'8321''45's_1942 = erased
-- Eval.ZeroBut.zbs-sum-s
d_zbs'45'sum'45's_1964 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'sum'45's_1964 = erased
-- Eval.ZeroBut.zb-zbs
d_zb'45'zbs_2000 ::
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
d_zb'45'zbs_2000 = erased
-- Eval.ZeroBut.zbs-sym
d_zbs'45'sym_2036 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'sym_2036 = erased
-- Eval.ZeroBut.zb-sym
d_zb'45'sym_2084 ::
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
d_zb'45'sym_2084 = erased
-- Eval.ZeroBut.zbs-cong
d_zbs'45'cong_2144 ::
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
d_zbs'45'cong_2144 = erased
-- Eval.ZeroBut.zb-sum
d_zb'45'sum_2186 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
   MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zb'45'sum_2186 = erased
-- Eval.ZeroBut.zbs-ext
d_zbs'45'ext_2214 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zbs'45'ext_2214 = erased
-- Eval.ZeroBut.zb-zbs-k
d_zb'45'zbs'45'k_2254 ::
  MAlonzo.Code.Real.T_Real_2 ->
  MAlonzo.Code.Real.T_RealProp_68 ->
  [Integer] ->
  [Integer] ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 ->
  (MAlonzo.Code.Data.List.Relation.Unary.All.T_All_44 -> AgdaAny) ->
  MAlonzo.Code.Agda.Builtin.Equality.T__'8801'__12
d_zb'45'zbs'45'k_2254 = erased
