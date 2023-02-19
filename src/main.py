from copy import deepcopy
import numpy as np
import random
from infotuple import selection_algorithms, metric_learners, body_metrics


def create_similarity_matrix(num_rows):
    img_similarities = [[random.random() for _ in range(num_rows)] for _ in range(num_rows)]
    print("Image similarity initial matrix: {0}".format(img_similarities))
    return np.array(img_similarities)


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
    num_burn_ins = 10
    num_iterations = 200
    tuple_size = 3  # 5 choices
    stim_list = [0, 1, 2, 3, 4]
    image_similarity_matrix = create_similarity_matrix(len(stim_list))  # numImages == (len(stim_list)^2)/2
    target_similarity_matrix = [
        [100, 50, 30, 20, 10],  # 0
        [50, 100, 70, 60, 80],  # 1
        [30, 70, 100, 40, 90],  # 2
        [20, 60, 40, 100, 55],  # 3
        [10, 80, 90, 55, 100]   # 4
    ]

    hundred_matrix = [
        [100, 100, 100, 100, 100],  # 0
        [100, 100, 100, 100, 100],  # 1
        [100, 100, 100, 100, 100],  # 2
        [100, 100, 100, 100, 100],  # 3
        [100, 100, 100, 100, 100]  # 4
    ]

    output_similarity_matrix = selection_algorithms.selection_algorithm(image_similarity_matrix,
                                                                        num_burn_ins,
                                                                        num_iterations,
                                                                        subject_oracle(target_similarity_matrix),
                                                                        metric_learners.probabilistic_mds,
                                                                        body_metrics.primal_body_selector,
                                                                        tuple_size,
                                                                        verbose_output=False)

    output_matrix = deepcopy(output_similarity_matrix)
    for row_num, row in enumerate(output_similarity_matrix):
        for row_num_2 in range(output_similarity_matrix.shape[0]):
            dist = np.linalg.norm(output_similarity_matrix[row_num_2] - row)
            output_matrix[row_num][row_num_2] = dist

    print(output_matrix)

    print(np.corrcoef((np.subtract(hundred_matrix, target_similarity_matrix).flatten()), output_matrix.flatten()))

    """
    k_matrix = np.dot(output_similarity_matrix, output_similarity_matrix.T)
    print(k_matrix)

    # generate new matrix
    distance_matrix = deepcopy(k_matrix)

    for row_num, row in enumerate(k_matrix):
        for col_num, cell_value in enumerate(row):
            distance = k_matrix[row_num][row_num] - 2 * cell_value + \
                       k_matrix[col_num][col_num]
            distance_matrix[row_num][col_num] = distance

    print(distance_matrix)
    print(distance_matrix/distance_matrix.std())
    """


if __name__ == "__main__":
    main()
