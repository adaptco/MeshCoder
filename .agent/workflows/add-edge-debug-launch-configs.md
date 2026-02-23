---
description: feat: Add VS Code launch configurations for Microsoft Edge debugging
---

# Add VS Code Launch Configurations for Microsoft Edge Debugging

This workflow adds the standard set of VS Code launch configurations for debugging with Microsoft Edge (Edge DevTools extension).

## Prerequisites

- VS Code with the [Microsoft Edge DevTools](https://marketplace.visualstudio.com/items?itemName=ms-edgedevtools.vscode-edge-devtools) extension installed
- A workspace folder open in VS Code

## Steps

1. Ensure the `.vscode/` directory exists at the root of the project:
```bash
mkdir -p .vscode
```

2. Create or update `.vscode/launch.json` with the following content. **Important**: Replace the `url` values with the correct path for your OS:
   - **Linux**: `/home/<username>/.antigravity-server/extensions/ms-edgedevtools.vscode-edge-devtools-<version>-universal/out/startpage/index.html`
   - **Windows**: `c:\\Users\\<username>\\.antigravity\\extensions\\ms-edgedevtools.vscode-edge-devtools-<version>-universal\\out\\startpage\\index.html`

```json
{
    "configurations": [
        {
            "name": "Attach to Edge",
            "port": 9222,
            "request": "attach",
            "type": "msedge",
            "webRoot": "${workspaceFolder}"
        },
        {
            "type": "pwa-msedge",
            "name": "Launch Microsoft Edge",
            "request": "launch",
            "runtimeArgs": [
                "--remote-debugging-port=9222"
            ],
            "url": "<STARTPAGE_URL>",
            "presentation": {
                "hidden": true
            }
        },
        {
            "type": "pwa-msedge",
            "name": "Launch Microsoft Edge in headless mode",
            "request": "launch",
            "runtimeArgs": [
                "--headless",
                "--remote-debugging-port=9222"
            ],
            "url": "<STARTPAGE_URL>",
            "presentation": {
                "hidden": true
            }
        },
        {
            "type": "vscode-edge-devtools.debug",
            "name": "Open Edge DevTools",
            "request": "attach",
            "url": "<STARTPAGE_URL>",
            "presentation": {
                "hidden": true
            }
        }
    ],
    "compounds": [
        {
            "name": "Launch Edge Headless and attach DevTools",
            "configurations": [
                "Launch Microsoft Edge in headless mode",
                "Open Edge DevTools"
            ]
        },
        {
            "name": "Launch Edge and attach DevTools",
            "configurations": [
                "Launch Microsoft Edge",
                "Open Edge DevTools"
            ]
        }
    ]
}
```

3. Stage and commit the new launch configuration:
```bash
git add .vscode/launch.json
git commit -m "feat: Add VS Code launch configurations for Microsoft Edge debugging"
```

4. Push to the remote branch and open a PR:
```bash
git push origin HEAD
```

## Notes

- The `url` fields in the `pwa-msedge` and `vscode-edge-devtools.debug` configurations point to the Edge DevTools extension startpage. Update the path to match the installed extension version on your system.
- The **Attach to Edge** config (port 9222) can be used independently without the startpage URL.
- The `hidden: true` presentation flag hides the individual configs in the Run & Debug picker — they are used via the **compounds** instead.
- If you need to also configure `pathMapping` in your VS Code `settings.json`, add:
```json
"vscode-edge-devtools.pathMapping": {
    "/": "${workspaceFolder}"
}
```
