#!/usr/bin/env python
# usage: evaluaton_script.py [-h] [--tweeteval_path TWEETEVAL_PATH]
#                            [--predictions_path PREDICTIONS_PATH] [--task TASK]

# optional arguments:
#   -h, --help: show this help message and exit
#   --tweeteval_path: Path to TweetEval dataset
#   --predictions_path: Path to predictions files
#   --task: Use this to get single task detailed results
#           (emoji|emotion|hate|irony|offensive|sentiment|stance)
#


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TweetEval evaluation script.")

    parser.add_argument("--tweeteval_path", default="./datasets/", type=str, help="Path to TweetEval datasets")
    parser.add_argument("--predictions_path", default="./predictions/", type=str, help="Path to predictions files")
    parser.add_argument(
        "--task", default="", type=str, help="Indicate this parameter to get single task detailed results"
    )

    args = parser.parse_args()

    if args.task == "":
        all_tweeteval_results = {}

        # Results for each task
        for t in TASKS:
            args.task = t
            all_tweeteval_results[t], _ = single_task_results(args)

        # Print results (score=-1 if some results are missing)
        print(f"{'-'*30}")
        if is_all_good(all_tweeteval_results):
            tweeteval_final_score = sum(all_tweeteval_results.values()) / len(all_tweeteval_results.values())
        else:
            tweeteval_final_score = -1
        for t in TASKS:
            # Each score
            print(f"{t}: {all_tweeteval_results[t]}")
        # Final score
        print(f"{'-'*30}\nTweetEval Score: {tweeteval_final_score}")

    else:
        # Detailed results of one single task (--task parameter)
        tweeteval_resut, results = single_task_results(args)
        for k in results:
            print(k, results[k])
        print(f"{'-'*30}\nTweetEval Score ({args.task}): {tweeteval_resut}")
