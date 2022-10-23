import tensorflow as tf
from migration_stuff import get_db as get_db
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

itoa = {i:s for s,i in actions.items()}

def one_hot_action(index):
    return tf.one_hot(index, len(actions))

def get_user_events(user_id):
    # print(customer_no)
    customer_event = get_db.get_database('events').find({"customer": user_id}).sort("date", 1)
    df = pd.DataFrame(list(customer_event))
    df = df[["unix_timestamp", "action"]]
    return df


if __name__ == "__main__":
    with open("util_files/unique_customer.txt") as writer:
        i = 0
        
        N = torch.zeros((len(actions), len(actions)), dtype=torch.int32)
            
        itoa = {i:s for s,i in actions.items()}

        while True:
            print(i)
            line = writer.readline()
            if not line:
                break
            if i == 100:
                break
            user_id = int(line)
            action = get_user_events(user_id)["action"].to_numpy()

            for act1, act2 in zip(action, action[1:]):
                ix1 = actions[act1]
                ix2 = actions[act2]
                N[ix1, ix2] += 1
            i += 1

    plt.figure(figsize=(24,24))
    plt.imshow(N, cmap='Blues')
    for i in range(len(actions)):
        for j in range(len(actions)):
            chstr = itoa[i][:4] + itoa[j][:4]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')
plt.tight_layout()
plt.savefig("bigram-action-2.png")
plt.show()