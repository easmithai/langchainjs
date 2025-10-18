import { test } from "@jest/globals";

import { fileURLToPath } from "node:url";
import * as path from "node:path";

import { FileState, GoogleGenAI } from "@google/genai";
import { ChatGoogleGenerativeAI } from "../chat_models.js";

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
});

const apiKey = process.env.GOOGLE_API_KEY || "";
const ai = new GoogleGenAI({ apiKey });

let uploadedFile: Awaited<ReturnType<typeof ai.files.upload>>;

beforeAll(async () => {
  // Download video file and save in src/tests/data
  // curl -O https://storage.googleapis.com/generativeai-downloads/data/Sherlock_Jr_FullMovie.mp4
  const displayName = "Sherlock Jr. video";

  const filename = fileURLToPath(import.meta.url);
  const dirname = path.dirname(filename);
  const pathToVideoFile = path.join(dirname, "/data/Sherlock_Jr_FullMovie.mp4");

  uploadedFile = await ai.files.upload({
    file: pathToVideoFile,
    config: {
      displayName,
      mimeType: "video/mp4",
    },
  });

  const { name } = uploadedFile;

  // Poll getFile() on a set interval (2 seconds here) to check file state.
  let file = await ai.files.get({ name });
  while (file.state === FileState.PROCESSING) {
    // Sleep for 2 seconds
    await new Promise((resolve) => {
      setTimeout(resolve, 2_000);
    });
    file = await ai.files.get({ name });
  }

  const systemInstruction =
    "You are an expert video analyzer, and your job is to answer " +
    "the user's query based on the video file you have access to.";
  const cachedContent = await ai.caches.create({
    model: "models/gemini-1.5-flash-001",
    config: {
      displayName: "gettysburg audio",
      ttl: "300s",
      systemInstruction: {
        role: "system",
        parts: [{ text: systemInstruction }],
      },
      contents: [
        {
          role: "user",
          parts: [
            {
              fileData: {
                mimeType: uploadedFile.mimeType,
                fileUri: uploadedFile.uri,
              },
            },
          ],
        },
      ],
    },
  });

  model.useCachedContent(cachedContent);
}, 10 * 60 * 1000); // Set timeout to 10 minutes to upload file

test("Test Google AI", async () => {
  const res = await model.invoke(
    "Introduce different characters in the movie by describing " +
      "their personality, looks, and names. Also list the " +
      "timestamps they were introduced for the first time."
  );

  expect(res).toBeTruthy();
});
