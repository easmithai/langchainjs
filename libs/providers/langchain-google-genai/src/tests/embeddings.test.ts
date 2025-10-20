import { beforeEach, expect, jest, test } from "@jest/globals";
import { GoogleGenerativeAIEmbeddings } from "../embeddings.js";

const embedContentMock = jest.fn();

beforeEach(() => {
  embedContentMock.mockReset();
});

test("embedQuery forwards outputDimensionality to the SDK", async () => {
  embedContentMock.mockResolvedValueOnce({
    embeddings: [{ values: [0.1, 0.2] }],
  });

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: "test-key",
    outputDimensionality: 128,
  });

  (embeddings as unknown as { client: { models: { embedContent: typeof embedContentMock } } }).client = {
    models: {
      embedContent: embedContentMock,
    },
  };

  await embeddings.embedQuery("hello world");

  expect(embedContentMock).toHaveBeenCalledWith(
    expect.objectContaining({
      config: expect.objectContaining({ outputDimensionality: 128 }),
    })
  );
});

test("embedDocuments forwards outputDimensionality to the SDK", async () => {
  embedContentMock.mockResolvedValue({
    embeddings: [
      { values: [0.1, 0.2] },
      { values: [0.3, 0.4] },
    ],
  });

  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: "test-key",
    outputDimensionality: 64,
  });

  (embeddings as unknown as { client: { models: { embedContent: typeof embedContentMock } } }).client = {
    models: {
      embedContent: embedContentMock,
    },
  };

  await embeddings.embedDocuments(["doc1", "doc2"]);

  expect(embedContentMock).toHaveBeenCalledTimes(1);
  expect(embedContentMock).toHaveBeenCalledWith(
    expect.objectContaining({
      config: expect.objectContaining({ outputDimensionality: 64 }),
    })
  );
});
