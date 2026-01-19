import os from 'os';
import fs from 'fs';
import path from 'path';
import puppeteer from 'puppeteer-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';
import ImageQueue from './image-queue.js';

process.on('unhandledRejection', (err, origin) => {
    console.error('Unhandled rejection ........................');
    console.error(err.error || err);
    log(`Unhandled rejection: ${err.error || err}`);
    setTimeout(() => process.exit(10), 200);
});

process.on('uncaughtException', (err, origin) => {
    console.error('Uncaught exception ........................');
    console.error(err.error || err);
    log(`Uncaught exception: ${err.error || err}`);
    setTimeout(() => process.exit(12), 200);
});

puppeteer.use(StealthPlugin());

let logType = 'console';

const NOW = { YMD: new Date().toISOString().substring(0, 10) };

const ROOT_DIR =
    os.type() === 'Windows_NT'
        ? path.dirname(import.meta.url.replace(/^file:\/\/\//, ''))
        : path.dirname(import.meta.url.replace(/^file:\/\//, ''));

const LOG = {
    dir: ROOT_DIR,
    name: `${NOW.YMD}-node.log`,
    path: path.join(ROOT_DIR, `${NOW.YMD}-node.log`)
};

const imageQ = new ImageQueue();

imageQ.on('error', err => {
    log(`ImageQ Error: ${err}`);
});

imageQ.on('log', message => {
    log(`ImageQ: ${message}`);
});

main().catch(error => {
    console.error(error);
    log(`General error: ${error}`);
});

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const nowYMD = NOW.YMD;
    const now = new Date().toISOString().substring(0, 19).replace(/:/g, '-');
    const outDir = path.isAbsolute(args.output) ? args.output : args.outputLocal;
    const failsDir = path.join(outDir, 'fails');
    const streamDir = path.join(outDir, nowYMD);

    confirmDir(outDir);
    confirmDir(failsDir);
    confirmDir(streamDir);

    log(`
    Script arguments:

    ${JSON.stringify(args, null, 4)}`);

    imageQ.dateSubdir = args.dateSubdir;
    imageQ.threshold = parseInt(args.intervalMs);
    imageQ.output = args.output;

    const browserPath = puppeteer.executablePath();
    log(`Browser path: ${browserPath} -> exists: ${fs.existsSync(browserPath)}`);

    const browser = await puppeteer.launch({
        // executablePath: path.join(ROOT_DIR, 'chrome/linux-127.0.6533.88/chrome-linux64/chrome'),
        // executablePath: '/usr/bin/chromium-browser',
        args: [
            '--no-sandbox',
            '--window-size=1920,1080',
            '--disable-setuid-sandbox',
            '--disable-blink-features=AutomationControlled',
            '--disable-notifications',
            '--disable-dev-shm-usage'
        ],
        headless: 'shell'
        // headless: true
    });

    const version = await browser.version();

    log(`browser version: ${version}`);
    log(`navigation started`);

    /**
     * @type {import('puppeteer').Page}
     */
    let page;
    let tries = 0;

    while (true) {
        try {
            page = await browser.newPage();

            await page.setUserAgent(
                'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36'
            );

            await page.setViewport({ width: 1920, height: 1080 });

            await navigate(`https://kick.com/${args.streamer}`, page);

            log(`navigation success`);

            break;
        } catch (error) {
            tries++;

            log(`navigation failure ${tries}`);

            if (!fs.existsSync(failsDir)) {
                fs.mkdirSync(failsDir);
            }

            try {
                await page.screenshot({
                    path: path.join(failsDir, `${now}-${tries}.png`)
                });
            } catch (error) {
                log('failed to do failure screenshot');
            }

            try {
                await page.close();
            } catch (error) {
                log('failed to close page');
            }
        }

        if (tries >= args.failRetries) {
            log(`${args.failRetries} retries failed exiting.`);
            process.exit(2);
        }
    }

    await wait(1000);

    try {
        await page.locator('div ::-p-text(Offline)').setTimeout(5000).click();
        log(`Streamer ${args.streamer} is offline`);
        process.exit(0);
    } catch (error) {}

    tries = 0;

    while (true) {
        const { success, error, errorType } = await tryStartWatching(page);
        if (success) {
            log(`start watching: ${args.streamer}`);
            break;
        } else {
            log(`start watching error(attempt ${tries + 1}): ${error.message}`);
            tries++;
        }

        try {
            await page.reload({ waitUntil: 'domcontentloaded' });
            log(`page reloaded`);
        } catch (error) {
            log(`failed to reload page`);
            process.exit(4);
        }

        if (tries >= args.failRetries) {
            log(`${args.failRetries} retries failed to start watching.`);
            process.exit(5);
        }
    }

    if (args.pageLogs === 'yes') {
        page.on('console', msg => {
            const message = msg.text();
            log(`PAGE LOG: ${message}`);

            if (message.startsWith('Script error:')) {
                page.screenshot({
                    path: path.join(
                        failsDir,
                        `${new Date().toISOString().substring(0, 19).replace(/:/g, '-')}-error.png`
                    )
                });
            }
        });
    }

    // try {
    //     await page.locator('* ::-p-text(1080p60)').setTimeout(2000).click();
    //     log(`1080p60`);
    // } catch (error) {
    //     // log(`failed to activate high resolution 3`);
    // }

    await page.exposeFunction('saveFrame', imageData => {
        imageQ.push(imageData);
    });

    await page.evaluate(async () => {
        await new Promise(resolve => setTimeout(() => resolve(true), 700));

        const video = document.querySelector('video');

        // const events = [
        //     'play',
        //     'playing',
        //     'waiting',
        //     'seeking',
        //     'seeked',
        //     'ended',
        //     'loadedmetadata',
        //     'loadeddata',
        //     'canplay',
        //     'canplaythrough',
        //     'durationchange',
        //     'timeupdate',
        //     'pause',
        //     'ratechange',
        //     'volumechange',
        //     'suspend',
        //     'emptied',
        //     'stalled'
        // ];

        // events.forEach(eventName => {
        //     video.addEventListener(eventName, () => {
        //         console.log(`Video ${eventName} event occurred`);
        //     });
        // });

        if (!video) {
            console.log(`Cannot find video element.`);
            return;
        }

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        const captureFrame = async () => {
            // canvas.width = video.videoWidth;
            // canvas.height = video.videoHeight;
            canvas.width = 1920;
            canvas.height = 1080;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            const dataUrl = canvas.toDataURL('image/png');

            window.saveFrame(dataUrl);

            if (!video.paused && !video.ended) {
                requestAnimationFrame(captureFrame);
            }
        };

        console.log(
            `before start video -> video.videoWidth=${video.videoWidth}, video.videoHeight=${video.videoHeight}`
        );

        await video.play();

        console.log(
            `started video -> video.videoWidth=${video.videoWidth}, video.videoHeight=${video.videoHeight}`
        );

        captureFrame();
    });
}

/**
 *
 * @param {string} url
 * @param {import('puppeteer').Page} page
 */
async function navigate(url, page) {
    await page.goto(url, { waitUntil: 'domcontentloaded' });

    log('navigation step 1 passed');

    await page.mouse.move(rand(1, 1920), rand(1, 1080), { steps: rand(5, 12) });
    await wait(100);

    await page.mouse.move(rand(1, 1920), rand(1, 1080), { steps: rand(5, 12) });
    await wait(900);

    await page.mouse.move(rand(1, 1920), rand(1, 1080), { steps: rand(5, 12) });
    await wait(100);

    await page.mouse.move(rand(1, 1920), rand(1, 1080), { steps: rand(5, 12) });
    await wait(500);

    await page.mouse.move(rand(1, 1920), rand(1, 1080), { steps: rand(5, 12) });
    await page.waitForSelector('video', { timeout: 9000 });

    log('navigation step 2 passed');
}

/**
 *
 * @param {import('puppeteer').Page} page
 */
async function tryStartWatching(page) {
    try {
        await page.locator('button ::-p-text(I am 18+)').setTimeout(3000).click();
        log(`18+ confirmation passed`);
    } catch (error) {}

    try {
        await page.locator('button ::-p-text(Start Watching)').setTimeout(5000).click();
    } catch (error) {
        try {
            await page.locator('video').setTimeout(5000).click();
        } catch (error) {
            return { success: false, error, errorType: 1 };
        }
    }

    await wait(1000);

    try {
        await page.locator('video').setTimeout(5000).hover();
        await page.locator('video + .z-controls button:nth-of-type(4)').setTimeout(5000).click();
        return { success: true };
    } catch (error) {
        return { success: false, error, errorType: 2 };
    }
}

function parseArgs(args) {
    const snakeToCamel = str =>
        str
            .toLowerCase()
            .replace(/([-_][a-z])/g, group =>
                group.toUpperCase().replace('-', '').replace('_', '')
            );

    const params = {
        outputLocal: '',
        output: '',
        streamer: '',
        intervalMs: '',
        failRetries: '',
        pageLogs: 'no',
        dateSubdir: 'yes'
    };

    args.forEach(arg => {
        if (arg.startsWith('--')) {
            const [key, value] = arg.split('=');
            const prop = snakeToCamel(key.substring(2));
            if (typeof params[prop] !== 'undefined') {
                params[prop] = value.trim();
            }
        }
    });

    if (!params.streamer) {
        console.error(`Error: missing streamer argument`);
        printHelp();
        process.exit(1);
    }

    if (!(params.intervalMs > 0)) {
        // Take screenshot every one second by default
        params.intervalMs = 1000;
    }

    if (!(params.failRetries > 0)) {
        // Exit the script after failRetries to access the streamer page
        params.failRetries = 5;
    }

    if (!params.output) {
        params.output = path.join(ROOT_DIR, 'output', params.streamer);
    }

    params.outputLocal = path.join(ROOT_DIR, 'output', params.streamer);

    return params;
}

function wait(ms) {
    return new Promise(resolve => setTimeout(() => resolve(true), ms));
}

function rand(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function confirmDir(dir) {
    if (!fs.existsSync(dir)) {
        log(`${dir} - directory not found `);
        fs.mkdirSync(dir, { recursive: true });
        log(`${dir} - created `);
    } else {
        log(`${dir} - exists`);
    }
}

function log(message) {
    const line = `${new Date().toISOString().substring(11)} | ${message}\r\n`;
    fs.appendFileSync(LOG.path, line, 'utf8');
    // console.log(line);
}

function printHelp() {
    console.log(`
    node index.js --output=<string> --streamer=<string> --interval-ms=<int> --fail-retries=<int> --page-logs=<yes|no> --date-subdir=<yes|no>

        --output - output directory path where the screenshots will be saved

        --streamer - streamer name that will be used to scrape e.g. https://kick.com/<streamer>

        --interval-ms - interval in milliseconds that screenshot will be taken. If 2 screenshots per second are needed, then this value will be 500

        --fail-retries - max sequential fail retries to access the stream page

        --page-logs - optional flag to determine if puppeteer.page logs are logged

        --date-subdir - optional flag to place images in YYYY-MM-DD subdirectory if output is absolute path

    example:

    node index.js --output=/abs/path/screenshots --streamer=classybeef --interval-ms=100
    `);
}
