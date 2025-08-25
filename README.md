# NewLang

A tiny experimental language with a friendly C-like syntax, composable types, and first-class callbacks — implemented in Python. 
This README shows how to set it up, run example programs, and gives you a quick tour of the language.

> Status: **experimental**. Expect dragons 🐉 — but also fun.

---

## Quick start

### Requirements
- **Python 3.9+**
- Windows, macOS, or Linux shell

### Get the code & run an example

```powershell
# Windows PowerShell
cd path\to\NEWLANG\compiler
py -m nl ..\examples\helloworld.nl
```

```bash
# macOS / Linux
cd path/to/NEWLANG/compiler
py -m nl ../examples/helloworld.nl
```

You should see progress updates printed to the console and a final “All done.”

### Run a different program
Point the module runner at any `.nl` file:

```powershell
py -m nl path\to\your_program.nl
```

### Build-only (no execute)
Semantic-check without running:
```bash
py -m nl --check ../examples/helloworld.nl
```

### Use a custom standard library
By default, the compiler loads `std.nl` from the compiler folder. To use a different one:
```bash
py -m nl --std /absolute/path/to/std.nl ../examples/helloworld.nl
```

---

## Language overview (TL;DR)

### Entry point
```nl
// Entry points must be defined by __main__ otherwise they will not execute from the specified point
__main__ 
def foo() : void {
    std:println("Hello from NewLang\n");
}
```

### Globals & arrays
```nl
const array<char> msg = std:toArray("Hi!");
```

### Types (structs) and constructors
```nl
declare
type struct {
    int x; int y; int z;
    ctor(x, y, z);
} Vector3;

extend type Vector3:ctor(x, y, z) {
    this->x = x; this->y = y; this->z = z;
}
```

### Methods (extensions)
```nl
// int - at this stage represents a number, integer or float
extend type Vector3:length() -> int {
    finalize ( (this->x * this->x) + (this->y * this->y) + (this->z * this->z) ) / 1.0; // toy
}

// By default methods must be declared before they can be used.
// This is done so that you can implement predefined runtime methods, or implement different actions across platforms.
// You can also return a custom struct by replacing void with that type name

declare def foo() -> void;
declare def bar() -> int;

// Then extend by using:
extend def foo() : void { 
    // body
}
extend def bar() : int {
   finalize -1;
}

```

### Inheritance
```nl
declare
type struct from Vector3 {
    int tag;
    ctor(x, y, z, tag);
} LabeledVec;

extend type LabeledVec(x, y, z, tag) into base(x, y, z) {
}

// You can create implicit operators by doing the following 
extend type LabeledVec(x, y, z, tag) into base(x, y, z) {
    // This will implicitly convert the current instance to a newly typed one
    finalize this->x;
}
```

### Member access & method call
```nl
Vector3 a(3, 4, 0);
auto len = a->length();
```

### Loops and While statements
```nl

def foo() : int {
    auto i = 0;
    while (1) {
       // Inline condition prevent the mess of single condition if statements.
       // But a traditional if statement can be used, and the loop can end by using just 'end;'

       end i > 10;
       i = i + 1;
       
       // or: if statementss doesn't always require () to execute
       if i > 10 and i > 7 {
           end;
       }
   }

   while (i < 10) {
       i = i + 1;
       if (i == 6) or (i == 3) {
           finalize i; // finalizing here will break from the method completly
       }
   }

   finalize i;
}

```


### Delegates / callbacks
- Create a **delegate collection**: `new[](&)()`
- Add a **lambda wrapper**: `[captures](&)(def -> Ret { ... })`
- Invoke with `->call(...)`

```nl
auto onTick = new[](&)();
onTick->add([args](&)(def {
    std:println("tick\n");
}));
onTick->call(); // invokes all handlers
```

### Returning values
Use `finalize` in any `def`/lambda/method/ctor body to return a value.
```nl
def add(a, b) : int { finalize a + b; }
```

### Imports (auto-loaded)
Imports pull in other `.nl`/`.nlh` files. Paths are resolved **relative to the importing file** and loaded automatically — no CLI juggling required. You can alias if you want:

```nl
import { "../compiler/vec3.nl" };          // no alias, still works
import { "../math/vec3.nl" } as V3;         // optional alias
```

Symbols (types/methods/declared funcs) from imported modules are made available to the program. If you use an alias, qualified calls like `V3:foo()` are supported (for declared functions).

---

## Examples

### 1) Hello World
```nl
const array<char> hello = std:toArray("Hello, world!\n");

__main__ 
def foo() : void {
    std:println(hello);
}
```

Run:
```powershell
py -m nl ..\examples\hello.nl
```

### 2) Vector3 quick demo
`vec3.nl`:
```nl
declare
type struct {
    int x; 
    int y; 
    int z;
    ctor(x, y, z);
} Vector3;

extend type Vector3:ctor(x, y, z) {
    this->x = x; this->y = y; this->z = z;
}

extend type Vector3:add(o) -> Vector3 {
    finalize Vector3(this->x + o->x, this->y + o->y, this->z + o->z);
}
```

`main.nl`:
```nl
import { "./vec3.nl" };

__main__
def foo() : void {
    Vector3 a(1,2,3);
    Vector3 b(4,5,6);
    auto c = a->add(b);
    std:println(c); // prints a dict-like object for now
}
```

Run:
```bash
py -m nl ./main.nl
```

### 3) Long-running job with progress
See `examples/longrun.nl`. It shows:
- Custom `Task` & `TaskRunner` structs
- Periodic progress snapshots emitted to a delegate collection
- `spin(workFactor)` CPU loops to fake work (~5 minutes if you increase `workFactor`)

Run (Windows):
```powershell
py -m nl ..\examples\longrun.nl
```

---

## CLI reference

```
python3 -m nl [--check] [--std path/to/std.nl] <source.nl> [more_sources.nl ...]
```

- `--check` – parse + semantic-check only; do not execute
- `--std PATH` – override the standard library path (default: `compiler/std.nl`)
- `<source.nl>` – one or more source files to compile/execute
- **Automatic import loading**: any `import { "..." }` found in sources is resolved and loaded recursively

---

## Tips & gotchas

- **Char arrays** print as Python lists for now (e.g. `['H', 'i']`).
- The array keyword can be used standalone and is type unaware.
- Objects print as dict-like structures: `{'__type': 'Vector3', 'x': 1, 'y': 2, 'z': 3}`.
- Use `this->field` in ctors/methods. Calls are `obj->method(args)`.
- Returning early? Use `finalize value;` inside the body.
- If you see *Unknown type 'X' used*, ensure the file that declares `X` is imported or present in the sources.
- Paths in `import { "..." }` are resolved relative to the importing file.

---

## Project layout (suggested)

```
NEWLANG/
├─ compiler/
│  ├─ nl.py            # module entrypoint (python -m nl)
│  ├─ std.nl           # standard library
│  └─ ...
├─ examples/
│  ├─ hello.nl
│  ├─ longrun.nl
│  └─ vec3.nl
└─ README.md
```

---

## Contributing

- Bug reports & small PRs welcome (crash repros are gold).
- Feature ideas: open an issue with a tiny sample program using the new syntax.

---

## License

MIT (or your preferred license—update this section accordingly).
