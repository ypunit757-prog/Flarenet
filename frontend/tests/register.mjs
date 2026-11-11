// Lets `node --test` import the app's .jsx files: esbuild turns them into plain modules on the way in.
import { register } from "node:module";
register("./jsx-loader.mjs", import.meta.url);
