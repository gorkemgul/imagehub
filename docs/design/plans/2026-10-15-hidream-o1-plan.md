# HiDream-O1 implementation plan

1. Pin and inventory the author checkpoint and official source, including all safetensors headers.
2. Add the native pixel patch projection, timestep embedder, output projection, and the mixed-mask
   Qwen3-VL generation forward while preserving published tensor names.
3. Add unit tests for shapes, masks, and published header layout, plus exact random-weight parity
   tests against the pinned official implementation.
4. Add the text-to-image backend, Flow-UniPC scheduler, catalog and registry integration, and tiny
   end-to-end parity for guidance on and off, two seeds and non-square images.
5. Add the offline real-checkpoint suite, attribution and model documentation, run focused parity
   and repository checks, and leave real-checkpoint execution to the controller.

