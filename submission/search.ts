// submission/search.ts
// Connects to the Rust search engine over TCP (localhost:7878).
//
// Start the engine:  cd engine && cargo run --release
// Then score:        bun run score:small  /  bun run score:large

import net from "node:net";

const PORT = 7878;
let socket: net.Socket | null = null;
let buf = "";
let resolve: ((line: string) => void) | null = null;
let reject: ((err: Error) => void) | null = null;

function onData(data: Buffer) {
  buf += data.toString("utf8");
  const nl = buf.indexOf("\n");
  if (nl !== -1 && resolve) {
    const line = buf.slice(0, nl).trimEnd();
    buf = buf.slice(nl + 1);
    const r = resolve;
    resolve = reject = null;
    r(line);
  }
}

function readLine(): Promise<string> {
  const nl = buf.indexOf("\n");
  if (nl !== -1) {
    const line = buf.slice(0, nl).trimEnd();
    buf = buf.slice(nl + 1);
    return Promise.resolve(line);
  }
  return new Promise((res, rej) => { resolve = res; reject = rej; });
}

function send(msg: string) { socket!.write(msg + "\n", "utf8"); }

export async function setup(datasetPath: string): Promise<void> {
  await new Promise<void>((res, rej) => {
    socket = net.createConnection({ port: PORT, host: "127.0.0.1" }, res);
    socket.on("data", onData);
    socket.on("error", (err) => { reject?.(err); reject = resolve = null; });
  });

  if (await readLine() !== "READY") throw new Error("Expected READY");
  send(`LOAD ${datasetPath}`);
  if (await readLine() !== "LOADED") throw new Error("Expected LOADED");
}

export async function search(query: string): Promise<string[]> {
  send(query);
  return JSON.parse(await readLine()) as string[];
}

export async function cleanup(): Promise<void> {
  socket?.destroy();
  socket = null;
  buf = "";
  resolve = reject = null;
}

export default search;