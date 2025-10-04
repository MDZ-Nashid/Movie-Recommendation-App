from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath, FilePath


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='src/config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
        protected_namespaces=('settings_',)
    )

    model_path: DirectoryPath
    model_name: str
    data_file: FilePath
    movie_name: FilePath
    model_name_svd: str
    model_name_nmf: str


model_settings = ModelSettings()
