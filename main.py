# pyright: basic
import argparse

from task1 import task1
from task2 import task2

parser = argparse.ArgumentParser(
    prog="aml",
    description="Runs tasks",
)

parser.add_argument(
    "--tn", type=int, choices=[1, 2], help="Task 2 run - 1 or 2 - default = both"
)
args = parser.parse_args()

if args.tn == 1:
    task1.run_task1()
elif args.tn == 2:
    task2.run_task2()
else:
    print("Running both")
    task1.run_task1()
    task2.run_task2()
