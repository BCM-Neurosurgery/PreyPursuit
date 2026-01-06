export PYTHONPATH=/Users/user/PyCharmProjects/PacManMain

configs=(
    "/Users/user/PycharmProjects/PacManMain/ChangeOfMind/config/modfitYFC.yaml"
)


# Loop through configs and start each process in parallel
for config in "${configs[@]}"; do
    python ModelFitting.py --config "$config" --mode main1 &
done
