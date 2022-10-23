import tensorflow as tf
from migration_stuff import get_db as get_db
import pandas as pd
from model import user_behaviour
import numpy as np

actions = {
"activity":0,
"campaign_bonus":1,
"checkout":2,
"checkout_item":3,
"coupon_invalidate":4,
"coupon_redeem":5,
"coupon_unassign":6,
"daily_login":7,
"level_down":8,
"level_set":9,
"level_up":10,
"merge":11,
"opt_in":12,
"opt_out":13,
"point_add":14,
"point_fix":15,
"point_spend":16,
"point_sub":17,
"points_expired":18,
"profile":19,
"profile_completed":20,
"program_transfer":21,
"referral":22,
"referral_bonus":23,
"referral_points":24,
"refund":25,
"reward": 26
}

def one_hot_action(action):
    return tf.one_hot(actions[action], len(actions))

def get_label(action):
    return actions[action]

def parse_user_id(user_id):
    customer_event = get_db.get_database('events').find({"customer": user_id}).sort("date", 1)
    df = pd.DataFrame(list(customer_event))
    df = df[["unix_timestamp", "action"]]
    df["unix_timestamp"] = df["unix_timestamp"] / df["unix_timestamp"].max()
    #df["action"] = df["action"].map(one_hot_action)
    # label = df["action"].map(get_label)
    lab_out = tf.stack(df["action"].map(one_hot_action))
    out = tf.concat([df["unix_timestamp"].values[..., tf.newaxis], lab_out], axis = -1)
    return lab_out, out

def generate_sample():
    # print(customer_no)
    with open("util_files/unique_customer.txt", "r") as reader:
        while True:
            line = reader.readline()
            if not line:
                break

            user_id = int(line)
            
            lab_out, out = parse_user_id(user_id)
            
            ln = len(out)
            max_ln = 10
            if(max_ln >= ln):
                paddings = tf.constant([[max_ln - ln, 0,], [0, 0]])
                yield tf.pad(out[:-1], paddings, "CONSTANT"), tf.pad(lab_out[1:], paddings, "CONSTANT")

            else:
                out = out[:-1]
                lab_out  = lab_out[1:]
                yield out[-max_ln+1:, :], lab_out[-max_ln+1:, :]

def get_dataset():
    dataset = tf.data.Dataset.from_generator(generate_sample, output_types = (tf.float32, tf.int64))
    return dataset.batch(10, num_parallel_calls = 5).prefetch(1)

if __name__ == "__main__":
    dataset = get_dataset()
    model = user_behaviour(len(actions))
    
    EPOCHS = 10

    checkpoint_filepath = './checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True)

    model.fit(dataset, epochs = EPOCHS, callbacks=[model_checkpoint_callback])
        

