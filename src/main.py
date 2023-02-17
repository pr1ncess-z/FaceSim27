import numpy as np
import random
from infotuple import selection_algorithms, metric_learners, body_metrics


def create_image_similarity_matrix(num_rows):
    img_similarities = [[random.random()*10 for _ in range(num_rows)] for _ in range(num_rows)]
    print("Image similarity initial matrix: {0}".format(img_similarities))
    return np.array(img_similarities)


def is_mouse_pressed(test_mouse):
    for button in test_mouse.getPressed():
        if button != 0:
            return True
    return False


def subject_oracle(target_similarity_matrix):
    def oracle(candidate_tuple):
        """"
        print("\n\n================= NEW ROUND ====================")
        print("Target similarity matrix: {0}".format(target_similarity_matrix))
        print("Working with tuple {0}".format(candidate_tuple))
        """
        response = candidate_tuple[0]
        response_list = [candidate_tuple[0]]
        candidates = candidate_tuple[1:]
        # print("Candidates: {0}".format(candidates))

        c_to_score = {}
        for c in candidates:
            c_to_score[c] = target_similarity_matrix[response][c]
        # print("c to score: {0}".format(c_to_score))

        candidates = sorted(candidates, key=lambda candidate: c_to_score[candidate], reverse=True)
        # print("Candidates sorted: {0}".format(candidates))

        response_list.extend(candidates)
        # print("responding with: {0}".format(response_list))
        return tuple(response_list)

    return oracle


def main():
    # --- Run the experiment ---
    num_burn_ins = 5
    num_iterations = 20
    tuple_size = 5  # 5 choices
    stim_list = [0, 1, 2, 3, 4]
    image_similarity_matrix = create_image_similarity_matrix(len(stim_list))  # numImages == (len(stim_list)^2)/2
    target_similarity_matrix = [
        [100, 50, 30, 20, 10],  # 0
        [50, 100, 70, 60, 80],  # 1
        [30, 70, 100, 40, 90],  # 2
        [20, 60, 40, 100, 55],  # 3
        [10, 80, 90, 55, 100]   # 4
    ]

    output_similarity_matrix = selection_algorithms.selection_algorithm(image_similarity_matrix,
                                                                        num_burn_ins,
                                                                        num_iterations,
                                                                        subject_oracle(target_similarity_matrix),
                                                                        metric_learners.probabilistic_mds,
                                                                        body_metrics.primal_body_selector,
                                                                        tuple_size,
                                                                        verbose_output=False)

    print(output_similarity_matrix)


if __name__ == "__main__":
    main()
