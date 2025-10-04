from build_models.model_builder import ModelBuilderService


def main():
    mdl_bld_svc = ModelBuilderService()
    mdl_bld_svc.train_model_nmf()


if __name__ == '__main__':
    main()
