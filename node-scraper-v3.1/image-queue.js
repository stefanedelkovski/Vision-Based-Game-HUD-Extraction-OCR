import fs from 'fs';
import path from 'path';
import EventEmitter from 'events';

export default class ImageQueue extends EventEmitter {
    set threshold(value) {
        if (typeof value === 'number' && value >= 0 && isFinite(value)) {
            this.settings.threshold = value;
        }
    }

    set output(value) {
        this.settings.output = value;
    }

    set dateSubdir(value) {
        this.settings.dateSubdir = /^(yes|true|1)$/i.test(value);
    }

    constructor() {
        super();
        this.queue = [];
        this.processing = false;
        this.processed = 0;
        this.lastProcessed = 0;
        this.settings = {
            threshold: 0,
            output: '',
            dateSubdir: false
        };
    }

    push(imageData) {
        this.queue.push(imageData);
        this.process();
    }

    async process() {
        if (this.processing) return;

        this.processing = true;

        while (this.queue.length > 0) {
            try {
                this.writeOutput(this.queue.shift());
            } catch (error) {
                this.emit('error', error);
            }
        }

        this.processing = false;
    }

    writeOutput(imageData) {
        if (!this.settings.output) {
            return this.emit('error', new Error('Output is not set.'));
        }

        const now = Date.now();

        if (this.settings.threshold && now - this.lastProcessed < this.settings.threshold) {
            return;
        }

        const base64Data = imageData.replace(/^data:image\/png;base64,/, '');
        const binaryData = Buffer.from(base64Data, 'base64');

        if (this.settings.output === 'stdout') {
            this.writeStdout(binaryData);
        } else {
            this.writeFile(binaryData);
        }

        this.processed++;
        this.lastProcessed = now;
    }

    writeStdout(binaryData) {
        const encodedData = binaryData.toString('base64');

        const bufferLength = Buffer.alloc(4);
        bufferLength.writeUInt32BE(encodedData.length, 0);

        process.stdout.write(bufferLength);
        process.stdout.write(encodedData);

        this.emit('log', `Image send to stdout - ${this.processed}`);
    }

    writeFile(binaryData) {
        const baseDir = this.settings.dateSubdir
            ? path.join(this.settings.output, new Date().toISOString().substring(0, 10))
            : this.settings.output;

        if (!fs.existsSync(baseDir)) {
            fs.mkdirSync(baseDir, { recursive: true });
        }

        const imagePath = path.join(baseDir, `frame_${Date.now()}.png`);

        fs.writeFileSync(imagePath, binaryData);

        this.emit('log', `Image saved: ${imagePath}`);
    }
}
