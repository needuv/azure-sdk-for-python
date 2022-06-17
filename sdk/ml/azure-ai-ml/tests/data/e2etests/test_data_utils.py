import pytest
from typing import Dict
from pathlib import Path
from azure.ai.ml._ml_exceptions import DataException
from azure.ai.ml._utils.utils import load_yaml
from azure.ai.ml._utils._data_utils import validate_mltable_metadata, download_mltable_schema
from azure.ai.ml.constants import MLTABLE_SCHEMA_URL_FALLBACK

from ml_test import MlRecordedTest
from devtools_testutils import recorded_by_proxy


@pytest.fixture
def mltable_schema():
    return download_mltable_schema(mltable_schema_url=MLTABLE_SCHEMA_URL_FALLBACK)


@pytest.mark.unittest
class TestDataUtils(MlRecordedTest):
    @recorded_by_proxy
    def test_validate_mltable_metadata_schema(self, tmp_path: Path, mltable_schema: Dict):
        mltable_folder = tmp_path / "mltable_folder"
        mltable_folder.mkdir()
        tmp_metadata_file = mltable_folder / "MLTable"

        file_contents = """
            paths:
                - file: ./tmp_file.csv
            transformations:
                - read_delimited:
                    delimiter: ","
                    encoding: ascii
                    header: all_files_same_headers
        """
        tmp_metadata_file.write_text(file_contents)
        valid_metadata_dict = load_yaml(tmp_metadata_file)
        validate_mltable_metadata(mltable_metadata_dict=valid_metadata_dict, mltable_schema=mltable_schema)
        # no errors raised

        file_contents = """
            transformations:
                - read_delimited:
                    delimiter: ","
                    encoding: ascii
                    header: all_files_same_headers
        """
        tmp_metadata_file.write_text(file_contents)
        missing_paths_dict = load_yaml(tmp_metadata_file)
        with pytest.raises(DataException) as ex:
            validate_mltable_metadata(mltable_metadata_dict=missing_paths_dict, mltable_schema=mltable_schema)
        assert "'paths' is a required property" in ex.value.message

        file_contents = """
            paths:
                - file: ./tmp_file.csv
            transformations:
                - read_delimited:
                    delimiter: ","
                    encoding: unknownencoding
                    header: all_files_same_headers
        """
        tmp_metadata_file.write_text(file_contents)
        invalid_encoding_dict = load_yaml(tmp_metadata_file)
        with pytest.raises(DataException) as ex:
            validate_mltable_metadata(mltable_metadata_dict=invalid_encoding_dict, mltable_schema=mltable_schema)
        assert "unknownencoding" in ex.value.message
