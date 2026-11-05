import { test, expect } from '@playwright/test';
for (const path of ['/', '/play', '/puzzles', '/training', '/review']) {
 test(`renders ${path} without a runtime crash`, async ({page}, testInfo) => {
  const errors=[];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/api/**', route => route.fulfill({status:401,contentType:'application/json',body:'{"detail":"Not authenticated"}'}));
  await page.goto(path);
  await expect(page.locator('#root')).not.toBeEmpty();
  await expect(page.locator('a[href="/play"]').first()).toBeVisible();
  await page.screenshot({path:testInfo.outputPath('page.png'),fullPage:true});
  expect(errors).toEqual([]);
 });
}
