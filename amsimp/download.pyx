#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
#cython: language_level=3
"""
AMSIMP Download Utility
"""

# -----------------------------------------------------------------------------------------#

# Importing Dependencies.
import sys, shutil, os
import tempfile
import math
import urllib.request as ulib
import urllib.parse as urlparse

# -----------------------------------------------------------------------------------------#

cdef class Download:
    """
    AMSIMP Download Utility 

    Public domain by anatoly techtonik <techtonik@gmail.com>
    Also available under the terms of MIT license
    Copyright (c) 2010-2015 anatoly techtonik
    """
    __current_size = 0

    def win32_unicode_console(self):
        """
        Enable unicode output to Windows console.
        """
        import codecs
        from ctypes import WINFUNCTYPE, windll, POINTER, byref, c_int
        from ctypes.wintypes import BOOL, HANDLE, DWORD, LPWSTR, LPCWSTR, LPVOID

        original_stderr = sys.stderr

        # Output exceptions in this code to original_stderr, so that we can at least see them.
        def _complain(message):
            original_stderr.write(message if isinstance(message, str) else repr(message))
            original_stderr.write('\n')

        codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp65001' else None)

        try:
            GetStdHandle = WINFUNCTYPE(HANDLE, DWORD)(("GetStdHandle", windll.kernel32))
            STD_OUTPUT_HANDLE = DWORD(-11)
            STD_ERROR_HANDLE = DWORD(-12)
            GetFileType = WINFUNCTYPE(DWORD, DWORD)(("GetFileType", windll.kernel32))
            FILE_TYPE_CHAR = 0x0002
            FILE_TYPE_REMOTE = 0x8000
            GetConsoleMode = WINFUNCTYPE(BOOL, HANDLE, POINTER(DWORD))(("GetConsoleMode", windll.kernel32))
            INVALID_HANDLE_VALUE = DWORD(-1).value

            def not_a_console(handle):
                if handle == INVALID_HANDLE_VALUE or handle is None:
                    return True
                return ((GetFileType(handle) & ~FILE_TYPE_REMOTE) != FILE_TYPE_CHAR
                        or GetConsoleMode(handle, byref(DWORD())) == 0)

            old_stdout_fileno = None
            old_stderr_fileno = None
            if hasattr(sys.stdout, 'fileno'):
                old_stdout_fileno = sys.stdout.fileno()
            if hasattr(sys.stderr, 'fileno'):
                old_stderr_fileno = sys.stderr.fileno()

            STDOUT_FILENO = 1
            STDERR_FILENO = 2
            real_stdout = (old_stdout_fileno == STDOUT_FILENO)
            real_stderr = (old_stderr_fileno == STDERR_FILENO)

            if real_stdout:
                hStdout = GetStdHandle(STD_OUTPUT_HANDLE)
                if not_a_console(hStdout):
                    real_stdout = False

            if real_stderr:
                hStderr = GetStdHandle(STD_ERROR_HANDLE)
                if not_a_console(hStderr):
                    real_stderr = False

            if real_stdout or real_stderr:
                WriteConsoleW = WINFUNCTYPE(BOOL, HANDLE, LPWSTR, DWORD, POINTER(DWORD), LPVOID)(("WriteConsoleW", windll.kernel32))

                class UnicodeOutput:
                    def __init__(self, hConsole, stream, fileno, name):
                        self._hConsole = hConsole
                        self._stream = stream
                        self._fileno = fileno
                        self.closed = False
                        self.softspace = False
                        self.mode = 'w'
                        self.encoding = 'utf-8'
                        self.name = name
                        self.flush()

                    def isatty(self):
                        return False

                    def close(self):
                        self.closed = True

                    def fileno(self):
                        return self._fileno

                    def flush(self):
                        if self._hConsole is None:
                            try:
                                self._stream.flush()
                            except Exception as e:
                                _complain("%s.flush: %r from %r" % (self.name, e, self._stream))
                                raise

                    def write(self, text):
                        try:
                            if self._hConsole is None:
                                text = text.encode('utf-8')
                                self._stream.write(text)
                            else:
                                text = text.decode('utf-8')
                                remaining = len(text)
                                while remaining:
                                    n = DWORD(0)
                                    retval = WriteConsoleW(self._hConsole, text, min(remaining, 10000), byref(n), None)
                                    if retval == 0 or n.value == 0:
                                        raise IOError("WriteConsoleW returned %r, n.value = %r" % (retval, n.value))
                                    remaining -= n.value
                                    if not remaining:
                                        break
                                    text = text[n.value:]
                        except Exception as e:
                            _complain("%s.write: %r" % (self.name, e))
                            raise

                    def writelines(self, lines):
                        try:
                            for line in lines:
                                self.write(line)
                        except Exception as e:
                            _complain("%s.writelines: %r" % (self.name, e))
                            raise

                if real_stdout:
                    sys.stdout = UnicodeOutput(hStdout, None, STDOUT_FILENO, '<Unicode console stdout>')
                else:
                    sys.stdout = UnicodeOutput(None, sys.stdout, old_stdout_fileno, '<Unicode redirected stdout>')

                if real_stderr:
                    sys.stderr = UnicodeOutput(hStderr, None, STDERR_FILENO, '<Unicode console stderr>')
                else:
                    sys.stderr = UnicodeOutput(None, sys.stderr, old_stderr_fileno, '<Unicode redirected stderr>')
        except Exception as e:
            _complain("exception %r while fixing up sys.stdout and sys.stderr" % (e,))

    def filename_from_url(self, url):
        """
        :return: detected filename as unicode or None
        """
        fname = os.path.basename(urlparse.urlparse(url).path)
        if len(fname.strip(" \n\t.")) == 0:
            return None
        return fname

    def filename_from_headers(self, headers):
        """
        Detect filename from Content-Disposition headers if present.

        :param: headers as dict, list or string
        :return: filename from content-disposition header or None
        """
        if type(headers) == str:
            headers = headers.splitlines()
        if type(headers) == list:
            headers = dict([x.split(':', 1) for x in headers])
        cdisp = headers.get("Content-Disposition")
        if not cdisp:
            return None
        cdtype = cdisp.split(';')
        if len(cdtype) == 1:
            return None
        if cdtype[0].strip().lower() not in ('inline', 'attachment'):
            return None

        fnames = [x for x in cdtype[1:] if x.strip().startswith('filename=')]
        if len(fnames) > 1:
            return None
        name = fnames[0].split('=')[1].strip(' \t"')
        name = os.path.basename(name)
        if not name:
            return None
        return name

    def filename_fix_existing(self, filename):
        """
        Expands name portion of filename with numeric ' (x)' suffix to
        return filename that doesn't exist already.
        """
        dirname = u'.'
        name, ext = filename.rsplit('.', 1)
        names = [x for x in os.listdir(dirname) if x.startswith(name)]
        names = [x.rsplit('.', 1)[0] for x in names]
        suffixes = [x.replace(name, '') for x in names]
        # Filter suffixes that match ' (x)' pattern.
        suffixes = [x[2:-1] for x in suffixes
                    if x.startswith(' (') and x.endswith(')')]
        indexes  = [int(x) for x in suffixes
                    if set(x) <= set('0123456789')]
        idx = 1
        if indexes:
            idx += sorted(indexes)[-1]
        return '%s (%d).%s' % (name, idx, ext)

    def bar_thermometer(self, current, total, width=80):
        """
        Return thermometer style progress bar string. `total` argument
        can not be zero. The minimum size of bar returned is 3. Example:

            [..........            ]

        Control and trailing symbols (\r and spaces) are not included.
        See `bar_adaptive` for more information.
        """
        # number of dots on thermometer scale
        avail_dots = width-2
        shaded_dots = int(math.floor(float(current) / total * avail_dots))
        return '[' + '.'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'

    def bar_adaptive(self, current, total, width=80):
        """
        Return progress bar string for given values in one of three
        styles depending on available width:

            [..  ] downloaded / total
            downloaded / total
            [.. ]

        if total value is unknown or <= 0, show bytes counter using two
        adaptive styles:

            %s / unknown
            %s

        if there is not enough space on the screen, do not display anything.

        Returned string doesn't include control characters like \r used to
        place cursor at the beginning of the line to erase previous content.

        This function leaves one free character at the end of string to
        avoid automatic linefeed on Windows.
        """
        # Process special case when total size is unknown and return immediately.
        if not total or total < 0:
            msg = "%s / unknown" % current
            if len(msg) < width:
                return msg
            if len("%s" % current) < width:
                return "%s" % current

        min_width = {
        'percent': 4,
        'bar': 3,
        'size': len("%s" % total)*2 + 3,
        }
        priority = ['percent', 'bar', 'size']

        # Select elements to show.
        selected = []
        avail = width
        for field in priority:
            if min_width[field] < avail:
                selected.append(field)
                avail -= min_width[field]+1

        # Render.
        output = ''
        for field in selected:
            if field == 'percent':
                # Fixed size width for percentage.
                output += ('%s%%' % (100 * current // total)).rjust(min_width['percent'])
            elif field == 'bar':  # [. ]
                # Bar takes its min width + all available space.
                output += self.bar_thermometer(current, total, min_width['bar']+avail)
            elif field == 'size':
                # Size field has a constant width (min == max).
                output += ("%s / %s" % (current, total)).rjust(min_width['size'])

        selected = selected[1:]
        if selected:
            output += ' '

        return output

    def callback_progress(self, blocks, block_size, total_size, bar_function):
        """
        callback function for urlretrieve that is called when connection is
        created and when once for each block

        draws adaptive progress bar in terminal/console

        use sys.stdout.write() instead of "print,", because it allows one more
        symbol at the line end without linefeed on Windows

        :param blocks: number of blocks transferred so far
        :param block_size: in bytes
        :param total_size: in bytes, can be -1 if server doesn't return it
        :param bar_function: another callback function to visualize progress
        """
        global __current_size
    
        width = min(100, 80)

        if sys.version_info[:3] == (3, 3, 0):  # regression workaround
            if blocks == 0:  # first call
                __current_size = 0
            else:
                __current_size += block_size
            current_size = __current_size
        else:
            current_size = min(blocks*block_size, total_size)
        progress = bar_function(current_size, total_size, width)
        if progress:
            sys.stdout.write("\r" + progress)

    def detect_filename(self, url=None, out=None, headers=None, default="download.wget"):
        """
        Return filename for saving file. If no filename is detected from output
        argument, url or headers, return default (download.wget)
        """
        names = dict(out='', url='', headers='')
        if out:
            names["out"] = out or ''
        if url:
            names["url"] = self.filename_from_url(url) or ''
        if headers:
            names["headers"] = self.filename_from_headers(headers) or ''
        return names["out"] or names["headers"] or names["url"] or default

    def download(self, url, out=None, bar=True):
        """
        High level function, which downloads URL into tmp file in current
        directory and then renames it to filename autodetected from either URL
        or HTTP headers.

        :param bar: function to track download progress (visualize etc.)
        :param out: output filename or directory
        :return:    filename where URL is downloaded to
        """
        # Detect of out is a directory.
        outdir = None
        if out and os.path.isdir(out):
            outdir = out
            out = None

        # Get filename for temp file in current directory.
        prefix = self.detect_filename(url, out)
        (fd, tmpfile) = tempfile.mkstemp(".tmp", prefix=prefix, dir=".")
        os.close(fd)
        os.unlink(tmpfile)

        # Set progress monitoring callback.
        def callback_charged(blocks, block_size, total_size):
            self.callback_progress(
                blocks, block_size, total_size, bar_function=self.bar_adaptive
            )

        if bar:
            callback = callback_charged
        else:
            callback = None

        binurl = list(urlparse.urlsplit(url))
        binurl[2] = urlparse.quote(binurl[2])
        binurl = urlparse.urlunsplit(binurl)

        (tmpfile, headers) = ulib.urlretrieve(binurl, tmpfile, callback)
        filename = self.detect_filename(url, out, headers)
        if outdir:
            filename = outdir + "/" + filename

        # Add numeric ' (x)' suffix if filename already exists.
        if os.path.exists(filename):
            filename = self.filename_fix_existing(filename)
        shutil.move(tmpfile, filename)

        return filename
