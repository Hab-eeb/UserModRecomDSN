from recagent.task_a.predictor import build_task_a_predictor

predictor = build_task_a_predictor("configs/task_a.yaml")

users = predictor.list_users(limit=5)
print("Users:", users)

user_id = users[0]["user_id"]
items = predictor.list_eval_items_for_user(user_id, limit=5)
print("Items:", items)

result = predictor.predict(
    user_id=user_id,
    target_domain=items[0]["domain"],
    target_parent_asin=items[0]["parent_asin"],
    include_ground_truth=True,
)

print(result)