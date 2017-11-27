# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BLEU metric implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import subprocess
import tempfile
import numpy as np

from six.moves import urllib
import tensorflow as tf


def accuracy(hypotheses, references, lowercase=False, ordered=False):
  """Calculate the accuracy for hypotheses and references.

  Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script

  Returns:
    The accuracy as a float32 value.
  """

  if np.size(hypotheses) == 0:
    return np.float32(0.0)

  total_accuracy = (np.asarray(hypotheses) == np.asarray(references)).sum() / float(len(references))

  tf.logging.info("Accuracy: %s", total_accuracy)

  return np.float32(total_accuracy)
