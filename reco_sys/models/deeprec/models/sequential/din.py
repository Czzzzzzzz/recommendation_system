import tensorflow as tf
from reco_sys.models.deeprec.models.sequential.sequential_base_model import (
     SequentialBaseModel,
)

__all__ = ["DIN_RECModel"]


class DIN_RECModel(SequentialBaseModel):
    """Deep Interest model

    :Citation:

    """

    def _build_seq_graph(self):
        """The main function to create din model.

        Returns:
            object: the output of din section.
        """
        hparams = self.hparams

        attention_mode = self.hparams.attention_mode

        with tf.compat.v1.variable_scope("din"):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2 # [batch_size, seq, embed]
            )
            self.mask = self.iterator.mask

            if hparams.attention_mode ==  "inner_product" or hparams.attention_mode == "outer_product":
                query = tf.expand_dims(self.target_item_embedding, axis=1) # [batch_size, 1, embed]
                self.weighted_hist_input = self._target_attention(query, hist_input, hist_input, attention_mode)
            elif hparams.attention_mode == "sum_pooling":    
                self.weighted_hist_input = tf.reduce_sum(hist_input, axis=1)
            else:
                raise ValueError("this attention mode not defined {0}".format(hparams.attention_mode))

            user_embed = self.weighted_hist_input
            model_output = tf.concat([user_embed, self.target_item_embedding], 1)

            tf.compat.v1.summary.histogram("model_output", model_output)
            return model_output

    def _target_attention(self, query, key, value, attention_mode="inner_product"):
        '''
            query: [batch_size, sequence=1, embed_size]
            key: [batch_size, sequence, embed_size]
            value: [batch_size, sequence, embed_size]
        '''
        hparams = self.hparams

        with tf.compat.v1.variable_scope("target_attention"):
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            if attention_mode == "inner_product":
                attention_logits = tf.math.reduce_sum(
                    tf.multiply(query, key), axis=-1
                ) # [batch_size, sequence]
            elif attention_mode == "outer_product":
                queries = tf.tile(query, (1, key.get_shape().as_list()[1], 1))
                din_input = tf.concat([queries, key, queries - key, tf.multiply(queries, key)], axis=-1)
                # din_input = tf.concat(queries, key, axis=-1)
                att_fnc_output = self._fcn_net(din_input, hparams.att_fcn_layer_sizes, scope="att_fcn")
                attention_logits = tf.squeeze(att_fnc_output, -1)

            mask_paddings = tf.ones_like(attention_logits) * (-(2 ** 32) + 1)
            if hparams.enable_softmax:
                attention_weights = tf.math.softmax(
                    tf.where(boolean_mask, attention_logits, mask_paddings), axis=-1
                ) #[batch_size, sequence]
            else:
                attention_weights = attention_logits
            weighted_sum = tf.reduce_sum(
                tf.multiply(tf.expand_dims(attention_weights, -1), value), axis=-2
            ) # [batch_size, emb]

        return weighted_sum