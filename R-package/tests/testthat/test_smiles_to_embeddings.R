test_that("smiles_to_embeddings works", {
  expect_error(smiles_to_embeddings("CCO"))
})
