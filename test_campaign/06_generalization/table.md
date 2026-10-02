| task | variant | status | n | acc [95% CI] | chance | conf | (acc-ch)/(1-ch) | d vs orig [95% CI] |
|---|---|---|---|---|---|---|---|---|
| cifar10 | orig(10 opt) | seen-task, held-out images | 80 | 0.963 [0.895,0.987] | 0.100 | 0.970 | 0.958 |  |
| cifar10 | 2 options | unseen option count | 80 | 1.000 [0.954,1.000] | 0.500 | 0.992 | 1.000 | +0.037 [+0.000,+0.087] |
| cifar10 | 3 options | unseen option count | 80 | 1.000 [0.954,1.000] | 0.333 | 0.985 | 1.000 | +0.037 [+0.000,+0.087] |
| cifar10 | 6 options | unseen option count | 80 | 0.975 [0.913,0.993] | 0.167 | 0.978 | 0.970 | +0.013 [+0.000,+0.037] |
| cifar10 | 8 options | unseen option count | 80 | 0.963 [0.895,0.987] | 0.125 | 0.971 | 0.957 | +0.000 [-0.037,+0.037] |
| cifar10 | synonym labels | novel vocabulary | 80 | 0.975 [0.913,0.993] | 0.100 | 0.949 | 0.972 | +0.013 [-0.025,+0.062] |
| cifar10 | 'a photo of a X' labels | novel vocabulary | 80 | 0.963 [0.895,0.987] | 0.100 | 0.973 | 0.958 | +0.000 [-0.037,+0.037] |
| cifar10 | letter MCQ keys A-J | novel format | 80 | 0.963 [0.895,0.987] | 0.100 | 0.970 | 0.958 | +0.000 [+0.000,+0.000] |
| cifar10 | digit keys | novel format | 80 | 0.963 [0.895,0.987] | 0.100 | 0.968 | 0.958 | +0.000 [+0.000,+0.000] |
| cifar10 | animal vs vehicle (new question) | novel question | 80 | 0.988 [0.933,0.998] | 0.500 | 0.939 | 0.975 | +0.025 [-0.025,+0.075] |
| cifar10 | living vs machine (new q, new words) | novel question | 80 | 0.912 [0.830,0.957] | 0.500 | 0.772 | 0.825 | -0.050 [-0.125,+0.025] |
| food101 | orig(20 opt) | seen-task, held-out images | 80 | 0.850 [0.756,0.912] | 0.050 | 0.838 | 0.842 |  |
| food101 | 3 options | unseen option count | 80 | 0.950 [0.878,0.980] | 0.333 | 0.945 | 0.925 | +0.100 [+0.025,+0.175] * |
| food101 | 8 options | unseen option count | 80 | 0.925 [0.846,0.965] | 0.125 | 0.915 | 0.914 | +0.075 [+0.000,+0.150] |
| food101 | 12 options | unseen option count | 80 | 0.863 [0.770,0.921] | 0.083 | 0.877 | 0.850 | +0.013 [-0.062,+0.087] |
| food101 | 30 options | unseen option count (>20) | 80 | 0.875 [0.785,0.931] | 0.033 | 0.798 | 0.871 | +0.025 [-0.062,+0.113] |
| food101 | letter MCQ, 4 opt | novel format | 80 | 0.950 [0.878,0.980] | 0.250 | 0.920 | 0.933 | +0.100 [+0.037,+0.175] * |
| food101 | letter MCQ, 20 opt | novel format | 80 | 0.850 [0.756,0.912] | 0.050 | 0.830 | 0.842 | +0.000 [-0.062,+0.062] |
| food101 | 'homemade X' labels, 8 opt | novel vocabulary | 80 | 0.900 [0.815,0.948] | 0.125 | 0.865 | 0.886 | +0.050 [-0.013,+0.125] |
| oxford_pets | orig(20 opt) | seen-task, held-out images | 80 | 0.650 [0.541,0.745] | 0.050 | 0.612 | 0.632 |  |
| oxford_pets | 3 options | unseen option count | 80 | 0.950 [0.878,0.980] | 0.333 | 0.813 | 0.925 | +0.300 [+0.188,+0.412] * |
| oxford_pets | 10 options | unseen option count | 80 | 0.750 [0.645,0.832] | 0.100 | 0.772 | 0.722 | +0.100 [+0.013,+0.188] * |
| oxford_pets | cat vs dog (new question) | novel question | 80 | 0.988 [0.933,0.998] | 0.500 | 0.980 | 0.975 | +0.338 [+0.237,+0.450] * |
| aokvqa | orig(4 opt) | seen-task, held-out images | 80 | 0.550 [0.441,0.654] | 0.250 | 0.471 | 0.400 |  |
| aokvqa | options shuffled | seen-task, perturbed | 80 | 0.525 [0.417,0.631] | 0.250 | 0.463 | 0.367 | -0.025 [-0.087,+0.025] |
| aokvqa | 2 options | unseen option count | 80 | 0.700 [0.592,0.789] | 0.500 | 0.667 | 0.400 | +0.150 [+0.050,+0.237] * |
| aokvqa | 3 options | unseen option count | 80 | 0.588 [0.478,0.689] | 0.333 | 0.548 | 0.381 | +0.037 [-0.037,+0.100] |
| aokvqa | keys w/x/y/z | novel format | 80 | 0.537 [0.429,0.643] | 0.250 | 0.453 | 0.383 | -0.013 [-0.062,+0.037] |
| scienceqa | orig(2-5 opt) | seen-task, held-out images | 80 | 0.512 [0.405,0.619] | 0.382 | 0.586 | 0.212 |  |
| scienceqa | keys w/x/y/z | novel format | 80 | 0.550 [0.441,0.654] | 0.382 | 0.566 | 0.272 | +0.037 [-0.037,+0.125] |
| eurosat | orig(10 opt) | seen-task, held-out images | 80 | 0.950 [0.878,0.980] | 0.100 | 0.982 | 0.944 |  |
| eurosat | synonym descriptions | novel vocabulary | 80 | 0.925 [0.846,0.965] | 0.100 | 0.953 | 0.917 | -0.025 [-0.062,+0.000] |
| eurosat | 3 options | unseen option count | 80 | 0.988 [0.933,0.998] | 0.333 | 0.991 | 0.981 | +0.037 [+0.000,+0.087] |
| coco | orig(4-8 opt, varies) | stage-1 only (not in stage-2 mix) | 80 | 0.637 [0.528,0.734] | 0.172 | 0.734 | 0.562 |  |
| coco | 2 options | stage-1 only; unseen option count | 80 | 0.887 [0.800,0.940] | 0.500 | 0.863 | 0.775 | +0.250 [+0.150,+0.362] * |
| coco | 3 options | stage-1 only; unseen option count | 80 | 0.850 [0.756,0.912] | 0.333 | 0.804 | 0.775 | +0.212 [+0.113,+0.325] * |
| coco | 10 options | stage-1 only; unseen option count | 80 | 0.625 [0.515,0.723] | 0.100 | 0.733 | 0.583 | -0.013 [-0.100,+0.087] |